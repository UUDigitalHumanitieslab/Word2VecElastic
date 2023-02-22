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

def test_hyphen_exception():
    analyzer = Analyzer('english', False).preprocess
    test_sentence = 'Our post offices are always open.'
    output = analyzer(test_sentence)
    assert len(output) == 3
    test_sentence = 'Post-war London was grim.'
    output = analyzer(test_sentence)
    assert 'post-war' in output
    test_sentence = 'We only accept anti anti-war sentiments'
    output = analyzer(test_sentence)
    assert 'anti' in output
    assert 'anti-war' in output
