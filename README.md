# GANcon: 

#### Protein contact map prediction with deep generative adversarial network



### Package dependencies:

- numpy

- tensorflow
- keras
- pyGaussDCA



### Install instructions:

```bash
python3 setup.py build
python3 setup.py install --user
```



### Usage instructions:

- Output a CASP format *rr* file:

```python
from gancon import *
model = gancon.get_model()
gancon.predict_rr(model,"1a2pA.fasta","1a2pA.aln","1a2pA.rr")
```



- Obtain a contact map prediction matrix: 

```python
from gancon import *
model = gancon.get_model()
A = gancon.predict(model,"1a2pA.fasta","1a2pA.aln")
print(A)
```

