from src.text_rules import has_text


def test_links_and_retweet_prefix_are_not_text():
    assert not has_text("https://t.co/abc")  # X image / video tweet
    assert not has_text("RT @WhiteHouse: https://t.co/abc")  # retweet of one
    assert not has_text("RT @a: !!")
    assert not has_text("RT: https://truthsocial.com/@x/123")  # Truth Social quote fallback
    assert not has_text("  ") and not has_text(None)


def test_short_words_over_a_link_are_text():
    assert has_text("Amen https://t.co/abc")
    assert has_text("RT @CENTCOM: Strikes on…")
    assert has_text("abc")
