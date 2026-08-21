from taiji import Taiji, TaijiConfig


def test_native_taiji_learns_a_raw_byte_cycle_online() -> None:
    data = b"abcdabcdabcdabcd"
    model = Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
    ))

    before = model.score_bytes(data)
    model.learn_bytes(data, epochs=200)
    after = model.score_bytes(data)

    assert after["mean_surprise"] < before["mean_surprise"]
    assert after["accuracy"] >= 0.75
    assert model.generate(b"a", 8) == b"bcdabcda"
