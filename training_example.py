"""
Simple training example using MiniTorch scalar autodiff.
"""

from minitorch.scalar import Scalar
from minitorch.datasets import simple

import random


class SimpleModel:
    """
    Logistic regression model:

        y_hat = sigmoid(w*x + b)
    """

    def __init__(self):

        self.w = Scalar(random.uniform(-1.0, 1.0))
        self.w.requires_grad_(True)

        self.b = Scalar(random.uniform(-1.0, 1.0))
        self.b.requires_grad_(True)

    

    def forward(self, x: float) -> Scalar:
        """Return prediction for input x."""
        return (self.w * x + self.b).sigmoid()

    def parameters(self):
        """Return trainable parameters."""
        return [self.w, self.b]

    def zero_grad(self):
        """Reset all parameter gradients."""
        for parameter in self.parameters():
            parameter.zero_grad_()


def binary_cross_entropy(pred: Scalar, target: float) -> Scalar:
    """
    Binary cross-entropy loss:

        L = -[y*log(p) + (1-y)*log(1-p)]
    """

    if target == 1:
        return -pred.log()

    return -(Scalar(1.0) - pred).log() # Scalar(1.0) because this operation belogns inside your Scalar computation graph


def train_simple(): #This is the fundamental machine-learning training loop.

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    X, y = simple(100)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SimpleModel()

    learning_rate = 0.5
    epochs = 100

    print(
        f"Initial parameters: "
        f"w={model.w.data:.4f}, "
        f"b={model.b.data:.4f}"
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(epochs):

        total_loss = 0.0
        correct = 0

        # Iteration over dataset 
        for (x_coord, _ ), label in zip(X, y):

            # Reset gradients
            model.zero_grad()

            # Forward
            pred = model.forward(x_coord)

            # Loss
            loss = binary_cross_entropy(pred, label)

            total_loss += loss.data

            # Backward
            loss.backward()

            # SGD update
            for parameter in model.parameters():

                if parameter.derivative is not None:

                    parameter.data -= (
                        learning_rate
                        * parameter.derivative
                    )

            # Accuracy
            predicted_label = (
                1 if pred.data >= 0.5 else 0
            )

            if predicted_label == label:
                correct += 1

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        average_loss = total_loss / len(X)
        accuracy = correct / len(X)

        if epoch % 10 == 0 or epoch == epochs - 1:


            print(
                f"Epoch {epoch:3d} | "
                f"Loss={average_loss:.4f} | "
                f"Accuracy={accuracy:.2%}"
            )

    # ---------------------------------------------------------
    # Final parameters
    # ---------------------------------------------------------

    print("\nTraining finished.")

    print(
        f"Final parameters: "
        f"w={model.w.data:.4f}, "
        f"b={model.b.data:.4f}"
    )

    print(
        "Expected decision boundary: "
        "x = 0.5"
    )

    if model.w.data != 0:

        learned_boundary = (
            -model.b.data / model.w.data
        )

        print(
            f"Learned decision boundary: "
            f"x = {learned_boundary:.4f}"
        )


if __name__ == "__main__":
    train_simple()



















            #           DATASET
            #          │
            #          ▼
            #   ┌─────────────┐
            #   │   x, y      │
            #   └──────┬──────┘
            #          │
            #          ▼
            #   ┌─────────────┐
            #   │   wx + b     │
            #   └──────┬──────┘
            #          │
            #          ▼
            #   ┌─────────────┐
            #   │   sigmoid    │
            #   └──────┬──────┘
            #          │
            #          ▼
            #     prediction
            #          │
            #          ▼
            #   ┌─────────────┐
            #   │     BCE      │
            #   └──────┬──────┘
            #          │
            #          ▼
            #        loss
            #          │
            #          ▼
            #   loss.backward()
            #          │
            #          ▼
            #   ┌─────────────┐
            #   │ MiniTorch   │
            #   │ Autograd    │
            #   └──────┬──────┘
            #          │
            #   ┌──────┴──────┐
            #   ▼             ▼
            # dL/dw          dL/db
            #   │             │
            #   └──────┬──────┘
            #          ▼
            #    SGD update
            #          │
            #          ▼
            #     new w, b
            #          │
            #          └──────────────┐
            #                         │
            #                         ▼
            #                   next sample