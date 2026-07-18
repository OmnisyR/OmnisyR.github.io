import unittest

from scripts.authoring_format import (
    extract_note_definitions,
    migrate_legacy_authoring,
    render_note_data,
)


class AuthoringFormatTests(unittest.TestCase):
    def test_extracts_semantic_note_definition(self):
        source = """Before.

<details class="omnisyr-note" lang="en">
<summary>Markov chain</summary>

The state is $x_t$.
</details>

After.
"""

        article, notes = extract_note_definitions(source)

        self.assertEqual(article.split(), ["Before.", "After."])
        self.assertEqual(
            notes,
            [
                {
                    "language": "en",
                    "term": "Markov chain",
                    "body": "The state is $x_t$.",
                }
            ],
        )

    def test_migrates_bilingual_legacy_note_to_semantic_details(self):
        source = """;;;a
;;;;;;;eMarkov chain::English note.;;;e;;;c马尔可夫链::中文注释。;;;c;;;;
;;;a
"""

        migrated = migrate_legacy_authoring(source)

        self.assertNotIn(";;;", migrated)
        self.assertIn('<details class="omnisyr-note" lang="en">', migrated)
        self.assertIn("<summary>Markov chain</summary>", migrated)
        self.assertIn("English note.", migrated)
        self.assertIn('<details class="omnisyr-note" lang="zh-CN">', migrated)
        self.assertIn("<summary>马尔可夫链</summary>", migrated)
        self.assertIn("中文注释。", migrated)

    def test_migrates_inline_language_pair_to_lang_spans(self):
        source = "## ;;;eDiffusion Models;;;e;;;c扩散模型;;;c"

        migrated = migrate_legacy_authoring(source)

        self.assertEqual(
            migrated,
            '## <span lang="en">Diffusion Models</span>'
            '<span lang="zh-CN">扩散模型</span>',
        )

    def test_renders_notes_as_structured_hidden_data(self):
        notes = [
            {
                "language": "en",
                "term": "Markov chain",
                "body": "State $x_t$.",
            }
        ]

        rendered = render_note_data(notes, lambda value: "<p>{}</p>".format(value))

        self.assertIn('class="omnisyr-denote-data" hidden', rendered)
        self.assertIn('<article lang="en">', rendered)
        self.assertIn("<h6>Markov chain</h6>", rendered)
        self.assertIn("<p>State $x_t$.</p>", rendered)
        self.assertNotIn(";;;", rendered)

    def test_separates_migrated_notes_from_following_language_block(self):
        source = """;;;a
;;;;Note::Text.;;;;
;;;a;;;e
## English
;;;e;;;c
## 中文
;;;c"""

        migrated = migrate_legacy_authoring(source)

        self.assertIn("</details>\n\n<div lang=\"en\">", migrated)


if __name__ == "__main__":
    unittest.main()
