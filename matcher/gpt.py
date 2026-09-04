from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI, BadRequestError

from matcher.models import Candidate, ProductRecord


ATTRIBUTE_NAMES = (
    "product_type_subtype",
    "form_format",
    "variant_model_product_line",
    "flavor_scent",
    "key_ingredient_material",
    "strength_concentration",
    "intended_use_function",
    "target_user_species_age",
    "compatibility_application",
    "dietary_processing_claims",
    "color_shade_pattern",
    "bundle_composition",
    "functional_size_package_configuration",
)
ATTRIBUTE_STATUSES = {"same", "different", "unknown", "not_applicable"}


class GptAdjudicator:
    def __init__(
        self,
        credentials_path: Path,
        cache_dir: Path,
        concurrency: int,
    ) -> None:
        config = yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
        config = config.get("openai", config)
        endpoint = config.get("endpoint")
        api_key = config.get("api_key")
        deployment = config.get("deployment_name")
        if not endpoint or not api_key or not deployment:
            raise ValueError(
                "Credentials YAML must contain endpoint, api_key, and deployment_name"
            )
        self.client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            max_retries=5,
        )
        self.deployment = deployment
        self.cache_dir = cache_dir / "gpt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        self._call_lock = asyncio.Lock()

    @staticmethod
    def _product_payload(record: ProductRecord) -> dict[str, Any]:
        return {
            "item_id": record.item_id,
            "name": record.product_type or record.normalized_name,
            "raw_name": record.raw_name,
            "brand": record.brand,
            "product_type": record.product_type,
            "category": list(record.category_path),
            "parsed_quantity": {
                "unit": record.quantity_unit,
                "total": record.total_quantity,
            },
        }

    async def adjudicate(
        self,
        a: ProductRecord,
        b_records: list[ProductRecord],
        candidates: list[Candidate],
        max_calls: int,
    ) -> int | None:
        offered = candidates[:20]
        payload = {
            "product_a": self._product_payload(a),
            "candidates_b": [
                {
                    **self._product_payload(b_records[candidate.b_index]),
                    "retrieval_sources": [
                        source
                        for source, present in (
                            ("exact_name", candidate.exact_name),
                            ("tfidf", candidate.lexical_score > 0),
                            ("embedding", candidate.vector_score > 0),
                        )
                        if present
                    ],
                }
                for candidate in offered
            ],
        }
        prompt = (
            "Choose whether one candidate is a realistic interchangeable consumer product. "
            "Products from different brands may match when their product type, variant, "
            "form, intended use, and other important attributes are equivalent. Brand "
            "difference alone is not a reason to reject a match. Different product types, "
            "flavors, forms, shades, or target animals do not match. "
            "Compare functional size and package configuration using the raw names and parsed "
            "quantity data. Treat modest differences in ordinary purchase quantity as same, "
            "such as 6 oz versus 8 oz or 18 count versus 20 count. Treat size or configuration "
            "as different when it changes functionality, compatibility, serving format, or "
            "intended use, such as 4-gallon versus 13-gallon bags, diaper size 3 versus size 4, "
            "or ten individual snack bags versus one family-size bag. Do not infer a conflict "
            "when one side omits the relevant size or package detail; use unknown instead. "
            "The normalized name may omit size text, so use raw_name and parsed_quantity for "
            "this comparison. For the selected "
            "candidate, compare every identity attribute listed below using exactly one "
            "status: same, different, unknown, or not_applicable. A match is forbidden if "
            "any attribute is different. Unknown means one side lacks enough information "
            "and is not itself a conflict. Bundle composition means whether the products "
            "contain the same kinds of items; functional_size_package_configuration covers "
            "size, capacity, count, dimensions, and package arrangement. Choose only an "
            "offered item_id or no_match. "
            "Return strict JSON with decision, selected_item_id_B, and attributes only. "
            "Do not include explanations or reasoning. "
            "decision must be either match or no_match. attributes must contain exactly "
            "these keys: "
            + ", ".join(ATTRIBUTE_NAMES)
            + ".\n"
            + json.dumps(payload, ensure_ascii=True)
        )
        key_material = f"{self.deployment}\n{prompt}"
        key = hashlib.sha256(key_material.encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            response = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache_hits += 1
        else:
            async with self._call_lock:
                if self.calls >= max_calls:
                    return None
                self.calls += 1
            async with self._semaphore:
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.deployment,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                except BadRequestError as exc:
                    body = exc.body if isinstance(exc.body, dict) else {}
                    error = body.get("error", body)
                    inner_error = error.get("innererror", {})
                    if (
                        error.get("code") != "content_filter"
                        and inner_error.get("code") != "ResponsibleAIPolicyViolation"
                    ):
                        raise
                    response = {
                        "decision": "no_match",
                        "selected_item_id_B": None,
                    }
                else:
                    content = completion.choices[0].message.content or "{}"
                    response = json.loads(content)
            cache_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

        decision = response.get("decision")
        selected_id = response.get("selected_item_id_B")
        allowed = {
            b_records[candidate.b_index].item_id: candidate.b_index for candidate in offered
        }
        if (
            decision != "match"
            or str(selected_id) not in allowed
        ):
            return None
        attributes = response.get("attributes")
        if not isinstance(attributes, dict) or set(attributes) != set(ATTRIBUTE_NAMES):
            return None
        if any(
            not isinstance(attributes.get(name), str)
            or attributes[name] not in ATTRIBUTE_STATUSES
            for name in ATTRIBUTE_NAMES
        ):
            return None
        if any(attributes[name] == "different" for name in ATTRIBUTE_NAMES):
            return None
        return allowed[str(selected_id)]
