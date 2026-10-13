"""Bounded symbolic pickle inspection: no imports, constructors or unpickling.

Only primitive containers and symbolic references are represented. REDUCE,
NEWOBJ and BUILD are recorded, never executed. Tensor/storage payloads stay opaque.
This extracts declared structure, not proof of what a malicious pickle will do.
"""
from dataclasses import dataclass
from io import BytesIO
import pickletools

MAX_BYTES = 64 * 1024 * 1024
MAX_OPS = 200_000
MAX_STRING = 512
UNKNOWN = object()
MARK = object()


@dataclass(frozen=True)
class Reference:
    module: str
    name: str


@dataclass
class Object:
    reference: Reference | None
    state: object = None


class InspectionError(ValueError):
    pass


def inspect_pickle(payload):
    """Return symbolic root and diagnostic; malformed/unsupported inputs fail closed."""
    if len(payload) > MAX_BYTES:
        return None, 'artifact_exceeds_inspection_limit'
    stack, memo = [], {}
    stream = BytesIO(payload)

    def take(count):
        if len(stack) < count or any(x is MARK for x in stack[-count:]):
            raise InspectionError()
        values = stack[-count:]
        del stack[-count:]
        return values

    def marked():
        values = []
        while stack and stack[-1] is not MARK:
            values.append(stack.pop())
        if not stack:
            raise InspectionError()
        stack.pop()
        return list(reversed(values))

    def pairs(target, values):
        if not isinstance(target, dict) or len(values) % 2:
            raise InspectionError()
        for key, value in zip(values[::2], values[1::2]):
            if isinstance(key, str):
                target[key] = value

    try:
        for count, (opcode, arg, _) in enumerate(pickletools.genops(stream), 1):
            if count > MAX_OPS:
                return None, 'pickle_operation_limit'
            name = opcode.name
            if name in {'PROTO', 'FRAME'}:
                continue
            if name == 'MARK':
                stack.append(MARK)
            elif name in {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
                          'STRING', 'BINSTRING', 'SHORT_BINSTRING'}:
                stack.append(arg if isinstance(arg, str) and len(arg) <= MAX_STRING else UNKNOWN)
            elif name in {'NONE', 'NEWTRUE', 'NEWFALSE', 'INT', 'BININT', 'BININT1', 'BININT2',
                          'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT'}:
                stack.append(None if name == 'NONE' else True if name == 'NEWTRUE' else False if name == 'NEWFALSE' else arg)
            elif name in {'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8', 'NEXT_BUFFER'}:
                stack.append(UNKNOWN)
            elif name == 'READONLY_BUFFER':
                if not stack:
                    raise InspectionError()
            elif name == 'GLOBAL':
                module, ref = arg.split(' ', 1)
                stack.append(Reference(module, ref) if len(arg) <= MAX_STRING else UNKNOWN)
            elif name == 'STACK_GLOBAL':
                module, ref = take(2)
                stack.append(Reference(module, ref) if isinstance(module, str) and isinstance(ref, str) else UNKNOWN)
            elif name in {'EXT1', 'EXT2', 'EXT4', 'PERSID'}:
                stack.append(UNKNOWN)
            elif name == 'BINPERSID':
                take(1)
                stack.append(UNKNOWN)
            elif name in {'PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE'}:
                if not stack:
                    raise InspectionError()
                memo[len(memo) if name == 'MEMOIZE' else arg] = stack[-1]
            elif name in {'GET', 'BINGET', 'LONG_BINGET'}:
                stack.append(memo[arg])
            elif name in {'EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_SET'}:
                stack.append([])
            elif name == 'EMPTY_DICT':
                stack.append({})
            elif name in {'LIST', 'TUPLE', 'FROZENSET'}:
                stack.append(marked())
            elif name in {'TUPLE1', 'TUPLE2', 'TUPLE3'}:
                stack.append(take(int(name[-1])))
            elif name == 'DICT':
                values = marked()
                target = {}
                pairs(target, values)
                stack.append(target)
            elif name in {'APPEND', 'APPENDS', 'ADDITEMS'}:
                values = take(1) if name == 'APPEND' else marked()
                target = stack[-1]
                if isinstance(target, list):
                    target.extend(values)
                elif not isinstance(target, Object):
                    raise InspectionError()
            elif name in {'SETITEM', 'SETITEMS'}:
                values = take(2) if name == 'SETITEM' else marked()
                target = stack[-1]
                if isinstance(target, dict):
                    pairs(target, values)
                elif not isinstance(target, Object):
                    raise InspectionError()
            elif name in {'NEWOBJ', 'NEWOBJ_EX', 'REDUCE'}:
                values = take(3 if name == 'NEWOBJ_EX' else 2)
                ref = values[0]
                # Protocol 0/1 instances often use copyreg._reconstructor.
                if (isinstance(ref, Reference) and ref.module in {'copyreg', 'copy_reg'}
                        and ref.name == '_reconstructor' and isinstance(values[1], list) and values[1]):
                    ref = values[1][0]
                if ref == Reference('collections', 'OrderedDict'):
                    # Only its empty-constructor form is interpreted; later SETITEMS populate it.
                    stack.append({} if values[1] == [] else UNKNOWN)
                else:
                    stack.append(Object(ref if isinstance(ref, Reference) else None))
            elif name == 'BUILD':
                state = take(1)[0]
                if not isinstance(stack[-1], Object):
                    raise InspectionError()
                stack[-1].state = state
            elif name == 'INST':
                marked()
                module, ref = arg.split(' ', 1)
                stack.append(Object(Reference(module, ref)))
            elif name == 'OBJ':
                values = marked()
                ref = values[0]
                stack.append(Object(ref if isinstance(ref, Reference) else None))
            elif name == 'POP':
                if not stack:
                    raise InspectionError()
                stack.pop()
            elif name == 'POP_MARK':
                marked()
            elif name == 'DUP':
                stack.append(stack[-1])
            elif name == 'STOP':
                if len(stack) != 1 or stack[0] is MARK or stream.tell() != len(payload):
                    raise InspectionError()
                return stack[0], None
            else:
                return None, 'unsupported_pickle_opcode'
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, UnicodeError):
        return None, 'malformed_or_unsupported_pickle'
    return None, 'incomplete_pickle'


def state_dict(value):
    if isinstance(value, Object):
        value = value.state
        # BUILD may hold (instance dict, slots dict); arbitrary root lists are not state.
        if isinstance(value, list) and len(value) == 2:
            value = {**(value[0] if isinstance(value[0], dict) else {}),
                     **(value[1] if isinstance(value[1], dict) else {})}
    return value if isinstance(value, dict) else {}
