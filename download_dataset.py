import datasets

data = datasets.load_dataset("turuta/Multi30k-uk", "flickr_2016")

data = data.rename_columns({"en": "sentence_eng_Latn", "uk": "sentence_ukr_Cyrl"})

data = data.map(with_indices=True, function=lambda x, i: {"id": i + 1, "sentence_eng_Latn": x["sentence_eng_Latn"], "sentence_ukr_Cyrl": x["sentence_ukr_Cyrl"]})
print(data["train"][0])

data = data.map(
    lambda x: {"text": f'[INST] {x["en"]} [/INST] {x["uk"]}'},
    remove_columns=["en", "uk"],
)

data = data["train"]

data = data.map(
    lambda x: {
        "text": x["text"]
        .replace(" ,", ",")
        .replace(" ,", ",")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace(" .", ".")
        .replace("–", "-")
        .replace(" '", "'")
        .replace(b' \xe2\x80\x8b'.decode('utf-8'), " ") # zero-width space
        .replace(' )', ')')
        .replace(" :", ":")
        .replace("` ", "`")
        .replace(b' \xe2\x80\x94'.decode('utf-8'), " -") # em dash
        # TODO: double check because of cases like "одна людина грає на барабанах, одна -на клавіатурі і одна людина- на гітарі."
        .replace("а- ", "а-")
        .replace("и- ", "и-")
        .replace("і- ", "і-")
        .replace("к- ", "к-")
        .replace("н- ", "н-")
        .replace("о- ", "о-") # TODO: special case with "Дво- і чотириногі сусіди заїжджають у гості."
        .replace("р- ", "р-")
        .replace("ю- ", "ю-")
        .replace(" -к", "-к")
        .replace(" -н", "-н")
        .replace(" -т", "-т")
        .replace(" -ц", " - ц")
        .replace("мотоциклі `.", "мотоциклі.`") # special case
        # TODO: remove different types of '"'
        .replace("«", '"')
        .replace("»", '"')
        .replace("„", '"')
        .replace("“", '"')
        .replace("”", '"')

        .replace("’", "'")
        .replace("`", "'")

    } 
)

data.to_json("multi-30k-uk.jsonl", force_ascii=False)

