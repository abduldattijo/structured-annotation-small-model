# Image alt text and sources (2026-10-23)

Tables stay as markdown in `draft.md`, which is the source. `make_figures.py` renders them to PNG for Medium and WordPress, so rerun it after any table edit.

| File | Alt text |
|---|---|
| featured_image.png | A small tray of blank cards sorted into color coded slots beside a large cabinet where several cards sit in the wrong slots or lie on the floor. |
| table1_test_accuracy.png | Table of test accuracy for four systems. The trained Qwen3 0.6B model averages 70.35 percent whole record accuracy, Haiku with label definitions 56.66, GPT-6 Sol 52.66, and Haiku with the original prompt 47.86. |
| table2_wrong_record_causes.png | Table counting wrong test records by first mistake for Qwen3 0.6B and Haiku. Qwen has 895 wrong records and Haiku 1,553. |
| table3_cost_and_speed.png | Table comparing Haiku and Qwen3 0.6B on cost per test set, one time cost, request speed, and GPU memory. |
| table4_result_to_action.png | Table with three columns, what happened, what it means, and what to do, with six rows drawn from the results. |

Featured image source: generated for this article with OpenAI gpt-image-1 (prompt in `generate_featured_image.py`). Say so in the caption. The image contains no text.

## Featured image (changed 2026-10-23)

The portal paused AI generated featured images, so the featured image is now `featured_chart.png`, drawn by `make_featured_chart.py` from the Table 1 means.

Alt text: Bar chart of whole record accuracy on 2,974 test commands. Qwen3 0.6B trained scores 70.35, Haiku with label definitions 56.66, GPT-6 Sol with the original prompt 52.66, and Haiku with the original prompt 47.86.

Caption: Chart of the Table 1 results. Image by author. The earlier AI generated picture `featured_image.png` is kept in the folder but is no longer used.
