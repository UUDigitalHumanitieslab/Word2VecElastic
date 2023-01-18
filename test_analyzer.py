from analyzer import Analyzer

def test_data_analyzer():
    analyzer = Analyzer('english', True).preprocess
    test_sentence = "What I'm saying is -- the U.K. have a neo-liberalist government!"
    output = analyzer(test_sentence)
    print(output)
    assert "neo-liberalist" in output
    assert "--" not in output
    assert "U.K." in output
    assert None not in output