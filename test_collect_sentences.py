from collect_sentences import es, DataCollector

from util import check_path

def test_data_collector(monkeypatch):
    def mock_search(index, body, size, scroll, track_total_hits):
        return {
            "_scroll_id": 42,
            "hits": {"total": {"value": 4},
            "hits": [
                {'_source': {"test_field": "This is a wonderful sentence. Here, have another."}},
                {'_source': {"test_field": "Nothing but GREAT sentences!"}},
                {'_source': {"test_field": "Creativity never ends!"}},
                {'_source': {"test_field": "I will not buy this record. It is scratched."}}
            ]}
        }
    
    def mock_clear_scroll(scroll_id):
        return {'acknowledged': True}

    monkeypatch.setattr(es, 'search', mock_search)
    monkeypatch.setattr(es, 'clear_scroll', mock_clear_scroll)
    
    check_path('test')
    sentences = DataCollector(
        'test_index',
        1982,
        1984,
        'english',
        'test_field',
        'test'
    )
    assert len(list(sentences)) == 12