from neurostasis import *

def test_main(capsys) -> None:
    main()
    stdout = capsys.readouterr().out
    assert "neurostasis" in stdout
