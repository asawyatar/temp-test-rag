# Retrieval cases

Generated from `retrieval_cases.jsonl` by `render_cases.py`; edit the JSONL.

Tenant A is Qamar Bay, tenant B is Rimal Terrace (see `fixtures/README.md`).

## Same language (13)

| ID | Tenant | Asked in | Question | Answer should come from |
|---|---|---|---|---|
| a-en-01 | A | en | What time does the pool open on Mondays? | `community-rules.en.md` |
| a-en-02 | A | en | How long does an emergency maintenance request take? | `maintenance.en.md` |
| a-en-03 | A | en | Which days is recycling collected? | `resident-faq.en.json` |
| a-en-04 | A | en | Where should we gather if there is a fire? | `safety-guide.en.pdf` |
| a-ar-05 | A | ar | كم رسوم الخدمات السنوية؟ | `fees-and-payments.ar.md` |
| a-ar-06 | A | ar | كم مدة صلاحية رمز دخول الزائر؟ | `visitors-and-access.ar.md` |
| a-ar-07 | A | ar | متى يجب أن أبلغ الإدارة برغبتي في تجديد العقد؟ | `lease-and-move-out.ar.docx` |
| b-ar-08 | B | ar | هل حجز ملعب التنس مجاني؟ | `facilities-booking.ar.md` |
| b-ar-09 | B | ar | كم تكلفة صيانة المكيف؟ | `maintenance.ar.md` |
| b-ar-10 | B | ar | متى تُجمع النفايات؟ | `resident-faq.ar.json` |
| b-ar-11 | B | ar | ما رقم تحويلة الأمن في حالة الحريق؟ | `safety-guide.ar.pdf` |
| b-en-12 | B | en | What is the late payment fee? | `fees-and-payments.en.md` |
| b-en-13 | B | en | How quickly is my deposit refunded after I move out? | `lease-and-move-out.en.docx` |

## Cross-language (11)

| ID | Tenant | Asked in | Question | Answer should come from |
|---|---|---|---|---|
| a-ar-14 | A | ar | هل يمكنني تربية كلب في الوحدة؟ | `community-rules.en.md` |
| a-ar-15 | A | ar | متى يُجرى تدريب الإخلاء من الحريق؟ | `safety-guide.en.pdf` |
| a-ar-16 | A | ar | كم تكلفة حجز القاعة المجتمعية لمناسبة خاصة؟ | `facilities-booking.en.md` |
| a-en-17 | A | en | Is there a fee if I pay my invoice late? | `fees-and-payments.ar.md` |
| a-en-18 | A | en | How many nights can a guest stay over? | `visitors-and-access.ar.md` |
| a-en-19 | A | en | What does it cost to end my lease early? | `lease-and-move-out.ar.docx` |
| b-en-20 | B | en | Is the gym open 24 hours? | `community-rules.ar.md` |
| b-en-21 | B | en | Can I install a satellite dish on my balcony? | `resident-faq.ar.json` |
| b-en-22 | B | en | How often are the smoke detectors checked? | `safety-guide.ar.pdf` |
| b-ar-23 | B | ar | هل يُجدَّد عقد الإيجار تلقائياً؟ | `lease-and-move-out.en.docx` |
| b-ar-24 | B | ar | كم عدد الزوار المسموح بدعوتهم في اليوم؟ | `visitors-and-access.en.md` |

## Shared app help (2)

| ID | Tenant | Asked in | Question | Answer should come from |
|---|---|---|---|---|
| a-en-25 | A | en | How do I add a family member to the app? | `app-help.en.md`, `app-help.ar.md`, `13-additional-users-and-directory.en.md`, `13-additional-users-and-directory.ar.md` |
| b-ar-26 | B | ar | كيف أغيّر لغة التطبيق؟ | `app-help.ar.md`, `app-help.en.md`, `16-profile-and-settings.ar.md`, `16-profile-and-settings.en.md` |

## Not in any document (2)

| ID | Tenant | Asked in | Question | Answer should come from |
|---|---|---|---|---|
| a-en-27 | A | en | When does the school bus pick up children? | nothing |
| b-ar-28 | B | ar | هل توجد شواحن للسيارات الكهربائية في المجمع؟ | nothing |

## Only the other tenant knows (2)

| ID | Tenant | Asked in | Question | Answer should come from |
|---|---|---|---|---|
| a-en-29 | A | en | Is the tennis court free to book? | nothing |
| | | | ↳ Only tenant_b has a tennis court. Tenant A's padel booking page is the honest nearest match; tenant B's tennis page must never appear. | |
| b-ar-30 | B | ar | كم سعر الساعة في ملعب البادل؟ | nothing |
| | | | ↳ Only tenant_a has a padel court. Tenant B's tennis booking is the nearest match; tenant A's padel page must never appear. | |
