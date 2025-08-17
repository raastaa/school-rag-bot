from routers.question_classifier import route_source


def test_director_route():
    assert route_source("как правильно составить приказ") == "director_handbook"
    assert route_source("другой вопрос") == "default"
