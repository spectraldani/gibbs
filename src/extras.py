import torch


def rgetattr(o, k_list):
    if len(k_list) == 0:
        return o
    else:
        return rgetattr(getattr(o, k_list[0]), k_list[1:])


def print_module(model, n_digits=10):
    print(f'{"parameter":35} {"device":7} {"dtype":7} {"shape":7}')
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            param_name = param_name.replace('raw_', '')
            param = rgetattr(model, param_name.split("."))
            print(
                f'{param_name:35} {param.device.type:7} {str(param.dtype)[6:]:7} '
                f'{str(tuple(param.shape)):7} {param.numpy().round(n_digits)}'
            )
