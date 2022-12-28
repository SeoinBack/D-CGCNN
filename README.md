# D-CGCNN : Direction-based Crystal Graph Convolutional Neural Netwokr

The D-CGCNN is a CGCNN (xie et al) based python code with direction-based crystal graph representation. 

The model.py is dropout-added graph convolutional neural network modified from original CGCNN by Juwhan Noh.  

For more details, check out this paper (https://pubs.acs.org/doi/pdf/10.1021/acs.chemmater.2c02498).

# Dependency

- Python3
- Numpy
- Pytorch
- Pymatgen
- Sklearn

# Usage

1. Clone thie repository:
<pre>
<code>
git clone https://github.com/SeoinBack/D-CGCNN
</code>
</pre>

2. Run example to test

<pre>
<code>
python main.py
python predict.py
</code>
</pre>

3. More details about inputs and arguments are same to original CGCNN, so you can found it in https://github.com/txie-93/cgcnn

# Publications 

If you use this code, please cite:

Dong Hyeon Mok, Jongseung Kim, Seoin Back. "Direction-Based Graph Representation to Accelerate Stable Catalyst
Discovery"
