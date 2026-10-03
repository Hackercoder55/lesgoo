"""See every Python operator any add-on runs.

``window_manager.operators`` only lists operators that register themselves for
undo/redo, so many add-on tools (pickers, pose libraries, AnimBot-style
helpers, studio tools) never show up there. Here every registered Python
operator class gets a thin wrapper around ``invoke``/``execute`` that logs the
call and then runs the original unchanged. Classes registered later (add-ons
enabled mid-session) are picked up by :func:`wrap_new_operators`.
"""

import functools

import bpy

from . import capture

WRAPPED = {}  # (class, method name) -> (original function, defined on the class itself)
EMIT = None


def _module_of(cls):
    module = getattr(cls, "__module__", "") or ""
    if module.startswith("bl_ext."):
        parts = module.split(".")
        return ".".join(parts[:3])
    return module.split(".")[0]


def _log_call(cls, method, operator):
    # invoke usually calls execute on the same instance: log that call once
    if getattr(operator, "_mouseadon_logged", False):
        return
    try:
        operator._mouseadon_logged = True
    except Exception:
        pass
    if EMIT is not None:
        EMIT(getattr(cls, "bl_idname", cls.__name__), getattr(cls, "bl_label", ""), _module_of(cls),
             method, capture.op_props(operator))


def _make_wrapper(cls, name, original):
    # Blender passes only as many arguments as the method's code declares
    # (co_argcount), so the wrapper must have exactly the operator signature.
    def log(self):
        try:
            _log_call(cls, name, self)
        except Exception:
            pass  # logging must never break the animator's tool

    if name == "invoke":
        def wrapper(self, context, event):
            log(self)
            return original(self, context, event)
    else:
        def wrapper(self, context):
            log(self)
            return original(self, context)

    functools.update_wrapper(wrapper, original)
    del wrapper.__wrapped__  # keep inspect.signature on the wrapper itself
    wrapper._mouseadon_original = original
    return wrapper


def _all_subclasses(base):
    seen = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        yield cls


def wrap_new_operators():
    """Wrap Python operator classes that are not wrapped yet. Returns count."""
    count = 0
    for cls in _all_subclasses(bpy.types.Operator):
        idname = getattr(cls, "bl_idname", "")
        if not idname or idname.startswith("mouseadon.") or not getattr(cls, "is_registered", False):
            continue
        for name in ("invoke", "execute"):
            if (cls, name) in WRAPPED:
                continue
            original = getattr(cls, name, None)
            if original is None or not callable(original) or not hasattr(original, "__code__"):
                continue  # only plain Python functions
            expected = 3 if name == "invoke" else 2
            if getattr(original, "_mouseadon_original", original).__code__.co_argcount != expected:
                continue  # unusual signature: leave it alone rather than risk breaking it
            owned = name in cls.__dict__
            original = getattr(original, "_mouseadon_original", original)  # inherited wrapper
            setattr(cls, name, _make_wrapper(cls, name, original))
            WRAPPED[(cls, name)] = (original, owned)
            count += 1
    return count


def unwrap_all():
    for (cls, name), (original, owned) in list(WRAPPED.items()):
        try:
            if owned:
                setattr(cls, name, original)
            elif name in cls.__dict__:
                delattr(cls, name)
        except Exception:
            pass
    WRAPPED.clear()
