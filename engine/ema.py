import torch


def copy_parameters(smodel: torch.nn.Module, tmodel: torch.nn.Module) -> None:
    for param in tmodel.parameters():
        param.detach_()
    sparams = list(smodel.parameters())
    tparams = list(tmodel.parameters())
    for i in range(0, len(sparams)):
        if not tparams[i].data.shape:
            tparams[i].data = sparams[i].data.clone()
        else:
            tparams[i].data[:] = sparams[i].data[:].clone()


def update_ema_model(
    iteration: int, alpha: float, model: torch.nn.Module, ema_model: torch.nn.Module
) -> None:
    # iteration shouldn't be 0
    with torch.no_grad():
        _alpha = min(1 - 1 / iteration, alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            if not ema_param.shape:
                ema_param.data = _alpha * ema_param.data + (1 - _alpha) * param.data
            else:
                ema_param.data[:] = (
                    _alpha * ema_param[:].data[:] + (1 - _alpha) * param[:].data[:]
                )
