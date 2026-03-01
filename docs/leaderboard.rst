Leaderboard Submission
======================

We maintain an `official leaderboard <https://huggingface.co/spaces/SyntheRela/leaderboard>`_ to benchmark synthetic relational data generation methods. To ensure fairness and reproducibility, **all evaluations are performed by the SyntheRela maintainers** on standardized hardware.

Evaluation Overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Specification
   * - **Compute**
     - Single NVIDIA H100 (80 GB)
   * - **Time Limit**
     - 48 hours execution time **per dataset**
   * - **Submission Frequency**
     - 1 submission per 30-day period
   * - **Capacity**
     - Up to 2 model variants/checkpoints per submission

How to Submit
-------------

1. **Prepare your code:** Ensure your method is reproducible and includes a clear ``README`` and ``requirements.txt``.
2. **Open an Issue:** Create a new `GitHub Issue <https://github.com/martinjurkovic/syntherela/issues>`_ using the title prefix ``[Model Submission]``.

For the complete requirements regarding environment setup, logging, and our privacy/confidentiality policy, please refer to the `Full Submission Guidelines <https://docs.google.com/document/d/1ae16L_vvT5PFt2OeN7FJauA_ayd_A6xCkhVJFoYcx04>`_.
