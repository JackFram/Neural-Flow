import torch
import torch.nn as nn
import torch.fx.symbolic_trace as symbolic_trace
from torch.fx.interpreter import *

from model import *


class funcInterpreter(Interpreter):
    def __init__(self, module: GraphModule, garbage_collect_values: bool = True):
        self.feature_list = []
        super().__init__(module, garbage_collect_values=garbage_collect_values)

    def run_node(self, n : Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n.target, n.name, args, kwargs)

    def placeholder(self, target: 'Target', name, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        if target.startswith('*'):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            ret = next(self.args_iter)
            self.feature_list.append((ret, name))
            return ret

    def call_method(self, target: 'Target', name, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(target, str)

        ret = getattr(self_obj, target)(*args_tail, **kwargs)
        self.feature_list.append((ret, name))
        return ret
        

    def call_function(self, target : Target, name, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        assert not isinstance(target, str)
        ret = target(*args, **kwargs)
        self.feature_list.append((ret, name))

        return ret

    def call_module(self, target: 'Target', name, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        ret = submod(*args, **kwargs)
        self.feature_list.append((ret, name))

        return ret
    
    def output(self, target: 'Target', name, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return super().output(target, args, kwargs)


def test():
    # model = VGG('VGG11')
    model = ResNet18()
    traced = symbolic_trace(model)
    # print(traced.graph)
    
    interp = funcInterpreter(traced)
    interp.run(torch.randn(1, 3, 32, 32))
    for feat in interp.feature_list:
        print(feat[1:])

if __name__ == "__main__":
    test()
