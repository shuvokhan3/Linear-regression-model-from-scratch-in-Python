from minitorch.module import Parameter, Module


def test_parameter():
    p = Parameter(10.0)

    assert p.value == 10.0
    assert p.shape == ()
    assert repr(p) == "Parameter(value=10.0)"

    p.update(20.0)

    assert p.value == 20.0


def test_parameter_shape():
    p = Parameter([1, 2, 3])

    # Python list has no .shape
    assert p.shape == ()


def test_module_initialization():
    module = Module()

    assert module._modules == {}
    assert module._parameters == {}
    assert module.training is True


def test_add_parameter():
    module = Module()

    p = module.add_parameter("weight", 5.0)

    assert isinstance(p, Parameter)
    assert p.value == 5.0

    assert "weight" in module._parameters
    assert module._parameters["weight"] is p


def test_add_existing_parameter():
    module = Module()

    p = Parameter(10.0)

    result = module.add_parameter("weight", p)

    assert result is p
    assert module._parameters["weight"] is p


def test_automatic_parameter_registration():
    module = Module()

    p = Parameter(10.0)

    module.weight = p

    assert "weight" in module._parameters
    assert module._parameters["weight"] is p
    assert module.weight is p


def test_automatic_module_registration():
    parent = Module()
    child = Module()

    parent.child = child

    assert "child" in parent._modules
    assert parent._modules["child"] is child
    assert parent.child is child


def test_modules():
    parent = Module()
    child1 = Module()
    child2 = Module()

    parent.child1 = child1
    parent.child2 = child2

    modules = parent.modules()

    assert modules == [child1, child2]


def test_modules_recursive():
    parent = Module()
    child = Module()
    grandchild = Module()

    parent.child = child
    child.grandchild = grandchild

    modules = parent.modules()

    assert modules == [child, grandchild]


def test_named_parameters():
    parent = Module()

    parent.weight = Parameter(1.0)
    parent.bias = Parameter(2.0)

    child = Module()
    child.weight = Parameter(3.0)
    child.bias = Parameter(4.0)

    parent.child = child

    named_params = parent.named_parameters()

    assert named_params == [
        ("weight", parent.weight),
        ("bias", parent.bias),
        ("child.weight", child.weight),
        ("child.bias", child.bias),
    ]


def test_named_parameters_nested():
    parent = Module()
    child = Module()
    grandchild = Module()

    parent.child = child
    child.grandchild = grandchild

    grandchild.weight = Parameter(5.0)

    named_params = parent.named_parameters()

    assert named_params == [
        ("child.grandchild.weight", grandchild.weight)
    ]


def test_parameters():
    parent = Module()

    parent.weight = Parameter(1.0)
    parent.bias = Parameter(2.0)

    child = Module()
    child.weight = Parameter(3.0)

    parent.child = child

    params = parent.parameters()

    assert params == [
        parent.weight,
        parent.bias,
        child.weight,
    ]


def test_train():
    parent = Module()
    child = Module()
    grandchild = Module()

    parent.child = child
    child.grandchild = grandchild

    parent.eval()

    assert parent.training is False
    assert child.training is False
    assert grandchild.training is False

    parent.train()

    assert parent.training is True
    assert child.training is True
    assert grandchild.training is True


def test_eval():
    parent = Module()
    child = Module()

    parent.child = child

    parent.eval()

    assert parent.training is False
    assert child.training is False