# Third-Party Notices

This repository contains code and aggregate result files for reproducible experiments on
repository-level code review evidence grounding.

## Scope of the Repository License

The repository `LICENSE` applies to the authors' own source code, experiment configuration files,
experiment scripts, and aggregate result CSV files included in this repository.

The repository license does not grant rights to third-party materials that are not owned by the
authors.

## Materials Not Redistributed

The repository does not redistribute:

- SWE-CARE source benchmark files, which users must obtain from the official source under the applicable upstream terms.
- CR-classification-ESEM23 source assets, which users must obtain from the official source under the applicable upstream terms.
- CodeReviewQA dataset files.
- Processed candidate-pool manifests derived from third-party benchmarks.
- Sample-level prediction files or error-analysis reports containing review comments, repository
  paths, or code snippets from third-party projects.
- Dense score caches and fine-tuned dense-model packages or weights. The authors' trained
  lightweight learned-ranker `model.json` is included under the repository's MIT License.
- Virtual environments or historical working archives.

Users who want to reproduce the processed splits must obtain the relevant benchmark files from
their official sources and comply with the terms of those datasets and upstream projects.

## External Dependencies

Python package dependencies are governed by their respective licenses.

## Release Scope

Only the research software, configuration, aggregate results, provenance documentation, and the
lightweight learned-ranker artifact listed in the release manifest are distributed here.
