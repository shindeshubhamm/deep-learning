class SGD:
    
    def __init__(self, params, lr=1e-3):
        super().__init__()
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()