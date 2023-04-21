from pit30m.torch.dataset import Pit30MLogDataset


def test_integration_init():
    dataset = Pit30MLogDataset(log_ids=["94f2e358-93cf-4d14-ea9f-9577a00c5fb0"])
    assert len(dataset) > 10
