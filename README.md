[![Build Status](https://travis-ci.com/orgoro/toxic-fool.svg?branch=master)](https://travis-ci.com/orgoro/toxic-fool)

# toxic-fool

## INTRODUCTION
A python package containing sub-packages divided into categories:
#### 1. Agents - _adversarial learned attck generators_
#### 2. Attacks - _optimization attacks like hot flip_
#### 3. Models - _attacked models_
#### 4. Data - _data handeling_
#### 5. Logs - _logs output_
#### 6. Resources - _resources for other categories_

## CODING STYLE
All code must be written according to [PEP8 coding style](https://www.python.org/dev/peps/pep-0008/).
Also we follow TensorFlow coding style where it doesn't contradicts PEP8 [TF coding style](https://www.tensorflow.org/community/style_guide)

## PYTHON COMPATIBILITY
All code needs to be compatible with Python 2 and 3.

Next lines should be present in all Python files:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```
Use [six](https://pypi.org/project/six/) to write compatible code (for example six.moves.range).

## PLATFORM COMPATIBILITY
We use os.path for paths
 
## ANNOTATIONS
For annotation we use the old syntax:

```python
def foo(mouse, traps):
    # type: (Mouse, list) -> Cheese
    cheese = do(mouse, traps)
    return cheese
```

Should be used in all interface functions i.e - functions exposed in category **\_\_init\_\_.py** file.

Read more: [PEP_484](https://www.python.org/dev/peps/pep-0484/).


## PACKAGES
Each package should contain:
```
1. __init__.py 
2. requirements.txt
3. README.md
4. example.py
5. test.py - using unittest library or tests under ./test
6. (optional) subfolders for:
    pkg_folder
        |-> src
        |-> test
        |-> data
        |-> scripts

```

## TESTING
* We use [py.test](https://docs.pytest.org/) framework for testing
* All test files and methods/functions start with the prefix **test_**
* Test names are *test\_when\_\<action\>\_then\_\<outcome\>*  
* Each test is comprised of 3 parts **arrange, act, assert** (3 a-s)
* Unittests should take less than **1 min** to run locally or marked as slowtest:

```python
import pytest
import unittest

class MyTestCase(unittest.TestCase):

    @pytest.mark.slow
    def test_my_slow_test(self):
        # ARRANGE:
        x = _arrange_x(seed=42)
        
        # ACT:
        result = act(x)
        
        # ASSERT:
        self.assertTrue(result)
```

* **Running slow tests**:`$ py.test --runslow`