# issue_test
## 1.problem description
  modle hava two outputs(out1,out2),there ere only a matmul operation between out1 and out2,as shown in the node diagram below<br>
both dynamic and static shape onnx modle output hava no loss of accuracy compare with torch modle output. when we run trt modle,<br>
there ere large loss of accuracy in dynamic shape trt modle output compare with torch output.
## 2.download modle
because github file size limit,you need download onnx and trt modle from url:https://drive.google.com/drive/folders/1EjL-bpaDkBBVlhZTSZSSKmv-9_zLpjVH,<br>
put all modle file to Same directory whit test script
## 3.onnx to trt command
  `static shape: trtexec --onnx=ttest1.onnx --workspace=300000 --saveEngine=t.plan --verbose --dumpLayerInfo --noTF32`<br>
  <br>
  `dynamic shape: trtexec --onnx=ttest1.onnx --minShapes=input_ids:1x512,bbox:1x512x4,images:1x3x224x224 --optShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224 --maxShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224   --workspace=300000 --saveEngine=t_mul.plan --verbose  --device=1 --dumpProfile`
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
