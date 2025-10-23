# GitHub Actions Workflows

## PyPI Publishing Workflow

The `publish-to-pypi.yml` workflow automates the process of publishing Python packages to PyPI and TestPyPI.

### How it works

The workflow is triggered when a new tag matching the pattern `v*` is pushed to the repository (e.g., `v0.1.0`, `v1.2.3`).

### Workflow steps

1. **Build**: Creates source distribution (`.tar.gz`) and wheel (`.whl`) packages
2. **Publish to TestPyPI**: Publishes the package to TestPyPI for testing
3. **Publish to PyPI**: Publishes the package to the official PyPI
4. **GitHub Release**: Creates a GitHub release and uploads signed artifacts

### Setup Requirements

Before the workflow can publish to PyPI, you need to configure trusted publishing:

#### For PyPI:
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher with:
   - **PyPI Project Name**: `syntherela`
   - **Owner**: `martinjurkovic`
   - **Repository name**: `syntherela`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `pypi`

#### For TestPyPI:
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new publisher with:
   - **PyPI Project Name**: `syntherela`
   - **Owner**: `martinjurkovic`
   - **Repository name**: `syntherela`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `testpypi`

### Usage

To trigger a release:

```bash
# Create and push a new tag
git tag v0.2.0
git push origin v0.2.0
```

The workflow will automatically:
- Build the distribution packages
- Publish to TestPyPI
- Publish to PyPI
- Create a GitHub release with signed artifacts

### Security

This workflow uses OpenID Connect (OIDC) trusted publishing, which is the recommended secure method for publishing to PyPI. No API tokens are required or stored in GitHub secrets.

All distribution packages are signed with Sigstore for enhanced security and verification.

### References

- [Official PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Sigstore](https://www.sigstore.dev/)
