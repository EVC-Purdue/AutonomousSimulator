from simnode.devices.slcan_pty import encode_std, encode_ext, parse_line


def test_slcan_roundtrip():
    d = bytes([1, 2, 3, 4, 0, 0, 0, 0])
    assert encode_std(0x100, d) == b"t1008" + d.hex().upper().encode() + b"\r"
    assert parse_line(encode_std(0x100, d)) == (0x100, False, d)
    assert parse_line(encode_ext(0x907, bytes(8))) == (0x907, True, bytes(8))
    assert parse_line(b"O\r") is None
