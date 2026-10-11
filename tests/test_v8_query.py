from v8.query import extract_one_question

def test_extracts_first_question_without_rewriting_semantics():
 x=extract_one_question('Which constraints matter most for this task? Here is an explanation. Another question?')
 assert x.valid and x.query=='Which constraints matter most for this task?'

def test_recovers_v7_json_query():
 x=extract_one_question('{"action":"ask","query":"What evidence should I provide?"}')
 assert x.valid and x.query=='What evidence should I provide?'

def test_stop_is_invalid_in_stage1():
 assert not extract_one_question('{"action":"stop"}').valid
