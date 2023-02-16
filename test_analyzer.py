from analyzer import Analyzer

def test_data_analyzer():
    analyzer = Analyzer('english', True).preprocess
    test_sentence = "What I'm saying is -- the U.K. have a neo-liberalist government!"
    output = analyzer(test_sentence)
    assert "neo-liberalist" in output
    assert "--" not in output
    assert "u.k." in output
    assert None not in output

def test_hyphen_merge():
    analyzer = Analyzer('english', True).preprocess
    test_sentence = 'The UK government are on the non-exciting but dangerous road of e-democracy.'
    output = analyzer(test_sentence)
    assert "non-exciting" in output
    assert "e-democracy" in output
    assert len(output) == 6
