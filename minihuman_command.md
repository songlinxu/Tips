
- Create virtual env: `python -m venv myenv`, `source myenv/bin/activate`
- `pip install --upgrade setuptools wheel build twine`
- You may need to update the pip
- `python setup.py sdist bdist_wheel`
- `twine upload dist/*`
