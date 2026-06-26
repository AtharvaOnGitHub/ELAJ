"""GradientReversalFunction: reverses gradient sign during backpropagation.

Used by DABClassifier to encourage study-invariant representations.
"""

import torch
from torch.autograd import Function
from torch import Tensor


class GradientReversalFunction(Function):
    """Custom autograd function that scales the gradient by -lambda.

    Forward pass: identity.
    Backward pass: multiply incoming gradient by -lambda_.

    The lambda_ parameter schedules adversarial training strength; a value of
    0.0 disables gradient reversal and a value of 1.0 gives full reversal.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None
