from pathlib import Path

from matcher.io import load_products


def test_load_products_with_pandas(tmp_path: Path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "item_id,name,brand_raw,item_info,sizing_comp\n"
        '1,"Goya Chick Peas, 16 oz",Goya,"{""category_0"":""Food""}","{}"\n',
        encoding="utf-8",
    )

    records = load_products(csv_path)

    assert len(records) == 1
    assert records[0].item_id == "1"
    assert records[0].brand == "goya"
    assert records[0].total_quantity == 16
