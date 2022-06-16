# issue_test
## 1.problem description
  modle hava two outputs(out1,out2),there ere only a matmul operation between out1 and out2,as shown in the node diagram below<br>
both dynamic and static shape onnx modle output hava no loss of accuracy compare with torch modle output. when we run trt modle,<br>
there ere large loss of accuracy in dynamic trt modle output compare with torch output.
## 2.download modle
because github file size limit,you need download onnx and trt modle from url:https://pan.baidu.com/s/1ItbXXFg_Wnu218D3IsJeRA?pwd=jfts,<br>
put modle file to Same directory whit test script
## 3.run onnx modle(dynamic shape and static shape)
  compare  Loss of accuracy between torch output and onnx(dynamic shape and static shape figure) output<br>
      `python onnx_run.py`<br>
## 4.run tensorrt modle(dynamic shape and static shape)
  compare loss Loss of accuracy between torch output and onnx(dynamic shape and static shape figure) output<br>
      `python trt_run.py`<br>
## 5.document introduction
out1.npy and out2.npy are torch modle output result
## 6.onnx output node
![image](https://user-images.githubusercontent.com/49616374/174082104-aa759e14-6e34-4b62-91ea-b6b7ea95f2f1.png)
