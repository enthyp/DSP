### Huffman encoder/decoder
Very basic implementation just to get the idea. 
  * works for first 5 paragraphs of _lorem ipsum_
  * no error handling

Running the example provided requires `graphviz` package.
It produces a following tree of occurrence counts that 
Huffman codewords are based on.

![alt-text](./out/tree.png)

As it turns out, the encoded representation uses 15786 
bits as opposed to 24008 required with ASCII encoding.
