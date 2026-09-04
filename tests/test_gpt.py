import asyncio
import json
from types import SimpleNamespace

import httpx
from openai import BadRequestError

from matcher.gpt import ATTRIBUTE_NAMES, GptAdjudicator
from matcher.models import Candidate
from matcher.normalize import normalize_product


class FakeCompletions:
    def __init__(self, differing_attribute: str | None = None) -> None:
        self.active = 0
        self.max_active = 0
        self.last_prompt = ""
        self.differing_attribute = differing_attribute

    async def create(self, **kwargs):
        self.last_prompt = kwargs["messages"][0]["content"]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        content = json.dumps(
            {
                "decision": "match",
                "selected_item_id_B": "b",
                "attributes": {
                    name: (
                        "different" if name == self.differing_attribute else "same"
                    )
                    for name in ATTRIBUTE_NAMES
                },
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class ContentFilteredCompletions:
    async def create(self, **kwargs):
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(400, request=request)
        body = {
            "code": "content_filter",
            "innererror": {"code": "ResponsibleAIPolicyViolation"},
        }
        raise BadRequestError("content filtered", response=response, body=body)


class MalformedAttributeCompletions(FakeCompletions):
    async def create(self, **kwargs):
        completion = await super().create(**kwargs)
        response = json.loads(completion.choices[0].message.content)
        response["attributes"]["flavor_scent"] = {"status": "same"}
        completion.choices[0].message.content = json.dumps(response)
        return completion


def product(item_id: str, name: str):
    return normalize_product({"item_id": item_id, "name": name})


def test_async_adjudication_respects_concurrency_and_call_limit(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=2)
    completions = FakeCompletions()
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    b_records = [product("b", "Black Beans 15 oz")]

    async def run_calls():
        requests = []
        for index in range(4):
            candidate = Candidate(
                b_index=0,
                lexical_score=0.7,
                vector_score=0.7,
            )
            requests.append(
                adjudicator.adjudicate(
                    product(f"a{index}", f"Black Beans 15 oz item {index}"),
                    b_records,
                    [candidate],
                    max_calls=3,
                )
            )
        return await asyncio.gather(*requests)

    results = asyncio.run(run_calls())

    assert adjudicator.calls == 3
    assert completions.max_active == 2
    assert sum(result is None for result in results) == 1
    assert "confidence" not in completions.last_prompt
    assert "Do not include explanations or reasoning" in completions.last_prompt
    assert "A match is forbidden if any attribute is different" in completions.last_prompt
    assert "size_within_tolerance" not in completions.last_prompt
    assert "functional_size_package_configuration" in completions.last_prompt
    assert "6 oz versus 8 oz" in completions.last_prompt
    assert "4-gallon versus 13-gallon bags" in completions.last_prompt
    assert "Brand difference alone is not a reason to reject" in completions.last_prompt


def test_adjudication_sends_full_retrieval_shortlist_without_local_scores(
    tmp_path,
) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=1)
    completions = FakeCompletions()
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    b_records = [product(f"b{index}", f"Black Beans Item {index}") for index in range(10)]
    candidates = [
        Candidate(
            b_index=index,
            lexical_score=0.9 if index < 5 else 0.0,
            vector_score=0.9 if index >= 5 else 0.0,
        )
        for index in range(10)
    ]

    asyncio.run(
        adjudicator.adjudicate(
            product("a", "Black Beans"),
            b_records,
            candidates,
            max_calls=1,
        )
    )

    payload = json.loads(completions.last_prompt.split("\n", 1)[1])
    assert len(payload["candidates_b"]) == 10
    assert all("local_score" not in candidate for candidate in payload["candidates_b"])
    assert payload["product_a"]["raw_name"] == "Black Beans"
    assert payload["product_a"]["parsed_quantity"] == {
        "unit": None,
        "total": None,
    }


def test_adjudication_sends_quantity_context(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=1)
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    b_records = [product("b", "Black Beans 15 oz")]
    candidate = Candidate(
        b_index=0,
        lexical_score=0.7,
    )

    result = asyncio.run(
        adjudicator.adjudicate(
            product("a", "Black Beans 30 oz"),
            b_records,
            [candidate],
            max_calls=1,
        )
    )

    assert result == 0
    assert adjudicator.calls == 1
    prompt = adjudicator.client.chat.completions.last_prompt
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["product_a"]["raw_name"] == "Black Beans 30 oz"
    assert payload["product_a"]["parsed_quantity"] == {
        "unit": "oz",
        "total": 30.0,
    }


def test_adjudication_rejects_match_with_attribute_conflict(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=1)
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(differing_attribute="flavor_scent")
        )
    )
    b_records = [product("b", "Shampoo Coconut")]
    candidate = Candidate(b_index=0, lexical_score=0.7)

    result = asyncio.run(
        adjudicator.adjudicate(
            product("a", "Shampoo Sandalwood"),
            b_records,
            [candidate],
            max_calls=1,
        )
    )

    assert result is None


def test_adjudication_rejects_malformed_attribute_status(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=1)
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=MalformedAttributeCompletions())
    )
    b_records = [product("b", "Shampoo Coconut")]
    candidate = Candidate(b_index=0, lexical_score=0.7)

    result = asyncio.run(
        adjudicator.adjudicate(
            product("a", "Shampoo Coconut"),
            b_records,
            [candidate],
            max_calls=1,
        )
    )

    assert result is None


def test_content_filter_is_treated_as_no_match_and_cached(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "openai:\n"
        "  endpoint: https://example.com/v1\n"
        "  api_key: test\n"
        "  deployment_name: test\n",
        encoding="utf-8",
    )
    adjudicator = GptAdjudicator(credentials, tmp_path, concurrency=1)
    adjudicator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=ContentFilteredCompletions())
    )
    b_records = [product("b", "Black Beans 15 oz")]
    candidate = Candidate(
        b_index=0,
        lexical_score=0.7,
    )

    first = asyncio.run(
        adjudicator.adjudicate(
            product("a", "Black Beans 15 oz"),
            b_records,
            [candidate],
            max_calls=1,
        )
    )
    second = asyncio.run(
        adjudicator.adjudicate(
            product("a", "Black Beans 15 oz"),
            b_records,
            [candidate],
            max_calls=1,
        )
    )

    assert first is None
    assert second == first
    assert adjudicator.calls == 1
    assert adjudicator.cache_hits == 1
