¹8
¿
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
7
Square
x"T
y"T"
Ttype:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¹4

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
: *
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0

input_dense2053/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
û
ß*'
shared_nameinput_dense2053/kernel

*input_dense2053/kernel/Read/ReadVariableOpReadVariableOpinput_dense2053/kernel* 
_output_shapes
:
û
ß*
dtype0

input_dense2053/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ß*%
shared_nameinput_dense2053/bias
z
(input_dense2053/bias/Read/ReadVariableOpReadVariableOpinput_dense2053/bias*
_output_shapes	
:ß*
dtype0

mid_dense991/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ßý*$
shared_namemid_dense991/kernel
}
'mid_dense991/kernel/Read/ReadVariableOpReadVariableOpmid_dense991/kernel* 
_output_shapes
:
ßý*
dtype0
{
mid_dense991/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ý*"
shared_namemid_dense991/bias
t
%mid_dense991/bias/Read/ReadVariableOpReadVariableOpmid_dense991/bias*
_output_shapes	
:ý*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	8*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	8*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0

mid_dense381/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ým*$
shared_namemid_dense381/kernel
|
'mid_dense381/kernel/Read/ReadVariableOpReadVariableOpmid_dense381/kernel*
_output_shapes
:	ým*
dtype0
z
mid_dense381/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*"
shared_namemid_dense381/bias
s
%mid_dense381/bias/Read/ReadVariableOpReadVariableOpmid_dense381/bias*
_output_shapes
:m*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
dtype0

mid_dense109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*$
shared_namemid_dense109/kernel
{
'mid_dense109/kernel/Read/ReadVariableOpReadVariableOpmid_dense109/kernel*
_output_shapes

:m*
dtype0
z
mid_dense109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namemid_dense109/bias
s
%mid_dense109/bias/Read/ReadVariableOpReadVariableOpmid_dense109/bias*
_output_shapes
:*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0

RMSprop/conv2d_6/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/conv2d_6/kernel/rms

/RMSprop/conv2d_6/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_6/kernel/rms*&
_output_shapes
:*
dtype0

RMSprop/conv2d_6/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv2d_6/bias/rms

-RMSprop/conv2d_6/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_6/bias/rms*
_output_shapes
:*
dtype0
¦
'RMSprop/batch_normalization_8/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'RMSprop/batch_normalization_8/gamma/rms

;RMSprop/batch_normalization_8/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_8/gamma/rms*
_output_shapes
:*
dtype0
¤
&RMSprop/batch_normalization_8/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/batch_normalization_8/beta/rms

:RMSprop/batch_normalization_8/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_8/beta/rms*
_output_shapes
:*
dtype0

RMSprop/conv2d_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameRMSprop/conv2d_7/kernel/rms

/RMSprop/conv2d_7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_7/kernel/rms*&
_output_shapes
: *
dtype0

RMSprop/conv2d_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_7/bias/rms

-RMSprop/conv2d_7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_7/bias/rms*
_output_shapes
: *
dtype0
¦
'RMSprop/batch_normalization_9/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'RMSprop/batch_normalization_9/gamma/rms

;RMSprop/batch_normalization_9/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_9/gamma/rms*
_output_shapes
: *
dtype0
¤
&RMSprop/batch_normalization_9/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&RMSprop/batch_normalization_9/beta/rms

:RMSprop/batch_normalization_9/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_9/beta/rms*
_output_shapes
: *
dtype0

RMSprop/conv2d_8/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_nameRMSprop/conv2d_8/kernel/rms

/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/kernel/rms*&
_output_shapes
: @*
dtype0

RMSprop/conv2d_8/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameRMSprop/conv2d_8/bias/rms

-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/bias/rms*
_output_shapes
:@*
dtype0
¨
(RMSprop/batch_normalization_10/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(RMSprop/batch_normalization_10/gamma/rms
¡
<RMSprop/batch_normalization_10/gamma/rms/Read/ReadVariableOpReadVariableOp(RMSprop/batch_normalization_10/gamma/rms*
_output_shapes
:@*
dtype0
¦
'RMSprop/batch_normalization_10/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'RMSprop/batch_normalization_10/beta/rms

;RMSprop/batch_normalization_10/beta/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_10/beta/rms*
_output_shapes
:@*
dtype0
¢
"RMSprop/input_dense2053/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
û
ß*3
shared_name$"RMSprop/input_dense2053/kernel/rms

6RMSprop/input_dense2053/kernel/rms/Read/ReadVariableOpReadVariableOp"RMSprop/input_dense2053/kernel/rms* 
_output_shapes
:
û
ß*
dtype0

 RMSprop/input_dense2053/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ß*1
shared_name" RMSprop/input_dense2053/bias/rms

4RMSprop/input_dense2053/bias/rms/Read/ReadVariableOpReadVariableOp RMSprop/input_dense2053/bias/rms*
_output_shapes	
:ß*
dtype0

RMSprop/mid_dense991/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ßý*0
shared_name!RMSprop/mid_dense991/kernel/rms

3RMSprop/mid_dense991/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense991/kernel/rms* 
_output_shapes
:
ßý*
dtype0

RMSprop/mid_dense991/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ý*.
shared_nameRMSprop/mid_dense991/bias/rms

1RMSprop/mid_dense991/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense991/bias/rms*
_output_shapes	
:ý*
dtype0

RMSprop/dense_6/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	8*+
shared_nameRMSprop/dense_6/kernel/rms

.RMSprop/dense_6/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_6/kernel/rms*
_output_shapes
:	8*
dtype0

RMSprop/dense_6/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_6/bias/rms

,RMSprop/dense_6/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_6/bias/rms*
_output_shapes
:*
dtype0

RMSprop/mid_dense381/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ým*0
shared_name!RMSprop/mid_dense381/kernel/rms

3RMSprop/mid_dense381/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense381/kernel/rms*
_output_shapes
:	ým*
dtype0

RMSprop/mid_dense381/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*.
shared_nameRMSprop/mid_dense381/bias/rms

1RMSprop/mid_dense381/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense381/bias/rms*
_output_shapes
:m*
dtype0
¨
(RMSprop/batch_normalization_11/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(RMSprop/batch_normalization_11/gamma/rms
¡
<RMSprop/batch_normalization_11/gamma/rms/Read/ReadVariableOpReadVariableOp(RMSprop/batch_normalization_11/gamma/rms*
_output_shapes
:*
dtype0
¦
'RMSprop/batch_normalization_11/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'RMSprop/batch_normalization_11/beta/rms

;RMSprop/batch_normalization_11/beta/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_11/beta/rms*
_output_shapes
:*
dtype0

RMSprop/mid_dense109/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*0
shared_name!RMSprop/mid_dense109/kernel/rms

3RMSprop/mid_dense109/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense109/kernel/rms*
_output_shapes

:m*
dtype0

RMSprop/mid_dense109/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/mid_dense109/bias/rms

1RMSprop/mid_dense109/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense109/bias/rms*
_output_shapes
:*
dtype0

RMSprop/output_layer/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!RMSprop/output_layer/kernel/rms

3RMSprop/output_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/output_layer/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/output_layer/bias/rms

1RMSprop/output_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_7/kernel/rms

.RMSprop/dense_7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/dense_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_7/bias/rms

,RMSprop/dense_7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_8/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_8/kernel/rms

.RMSprop/dense_8/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_8/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/dense_8/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_8/bias/rms

,RMSprop/dense_8/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_8/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_9/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_9/kernel/rms

.RMSprop/dense_9/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_9/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/dense_9/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_9/bias/rms

,RMSprop/dense_9/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_9/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
Ê
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¼É
value±ÉB­É B¥É
Æ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"
signatures*
* 
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
Õ
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
Õ
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses*

O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
¦

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
Õ
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
* 

h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses* 
¦

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses*

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
ª

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	 bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses*
¬
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«_random_generator
¬__call__
+­&call_and_return_all_conditional_losses* 
®
®kernel
	¯bias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses*
®
¶kernel
	·bias
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses*

¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses* 

Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses* 
®
Êkernel
	Ëbias
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses*
®
Òkernel
	Óbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses*
Ø
	Úiter

Ûdecay
Ülearning_rate
Ýmomentum
Þrho
#rmsþ
$rmsÿ
,rms
-rms
<rms
=rms
Erms
Frms
Urms
Vrms
^rms
_rms
nrms
orms
|rms
}rmsrmsrmsrmsrmsrmsrmsrms rms®rms¯rms¶rms·rmsÊrmsËrmsÒrmsÓrms*
Ì
#0
$1
,2
-3
.4
/5
<6
=7
E8
F9
G10
H11
U12
V13
^14
_15
`16
a17
n18
o19
|20
}21
22
23
24
25
26
27
28
29
30
 31
®32
¯33
¶34
·35
Ê36
Ë37
Ò38
Ó39*

#0
$1
,2
-3
<4
=5
E6
F7
U8
V9
^10
_11
n12
o13
|14
}15
16
17
18
19
20
21
22
 23
®24
¯25
¶26
·27
Ê28
Ë29
Ò30
Ó31*
R
ß0
à1
á2
â3
ã4
ä5
å6
æ7
ç8
è9* 
µ
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
* 

îserving_default* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
,0
-1
.2
/3*

,0
-1*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
E0
F1
G2
H3*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
^0
_1
`2
a3*

^0
_1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 
* 
* 
f`
VARIABLE_VALUEinput_dense2053/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEinput_dense2053/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

n0
o1*

ß0
à1* 
·
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
¡activity_regularizer_fn
*u&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 
c]
VARIABLE_VALUEmid_dense991/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmid_dense991/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

|0
}1*

á0
â1* 
º
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
­activity_regularizer_fn
+&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEmid_dense381/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmid_dense381/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

ã0
ä1* 
¼
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
	variables
trainable_variables
regularization_losses
__call__
¹activity_regularizer_fn
+&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_11/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUEmid_dense109/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmid_dense109/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*

å0
æ1* 
¼
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
Åactivity_regularizer_fn
+¦&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses* 
* 
* 
* 
d^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

®0
¯1*

®0
¯1*

ç0
è1* 
¼
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
Ñactivity_regularizer_fn
+µ&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_7/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

¶0
·1*

¶0
·1*
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_8/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ê0
Ë1*

Ê0
Ë1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_9/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_9/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ò0
Ó1*

Ò0
Ó1*
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
>
.0
/1
G2
H3
`4
a5
6
7*
Â
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*

ì0
í1
î2*
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

`0
a1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ß0
à1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

á0
â1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ã0
ä1* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 

å0
æ1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ç0
è1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

ïtotal

ðcount
ñ	variables
ò	keras_api*
M

ótotal

ôcount
õ
_fn_kwargs
ö	variables
÷	keras_api*
z
øtrue_positives
ùtrue_negatives
úfalse_positives
ûfalse_negatives
ü	variables
ý	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ï0
ð1*

ñ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ó0
ô1*

ö	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
ø0
ù1
ú2
û3*

ü	variables*

VARIABLE_VALUERMSprop/conv2d_6/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_6/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/batch_normalization_8/gamma/rmsSlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&RMSprop/batch_normalization_8/beta/rmsRlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_7/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_7/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/batch_normalization_9/gamma/rmsSlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&RMSprop/batch_normalization_9/beta/rmsRlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_8/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_8/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(RMSprop/batch_normalization_10/gamma/rmsSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/batch_normalization_10/beta/rmsRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"RMSprop/input_dense2053/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE RMSprop/input_dense2053/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense991/kernel/rmsTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense991/bias/rmsRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_6/kernel/rmsTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUERMSprop/dense_6/bias/rmsRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense381/kernel/rmsTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense381/bias/rmsRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(RMSprop/batch_normalization_11/gamma/rmsTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/batch_normalization_11/beta/rmsSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense109/kernel/rmsUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/mid_dense109/bias/rmsSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output_layer/kernel/rmsUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output_layer/bias/rmsSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_7/kernel/rmsUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_7/bias/rmsSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_8/kernel/rmsUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_8/bias/rmsSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_9/kernel/rmsUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_9/bias/rmsSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_6_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿU

%serving_default_input_dense2053_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿû

í

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_6_input%serving_default_input_dense2053_inputconv2d_6/kernelconv2d_6/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceinput_dense2053/kernelinput_dense2053/biasmid_dense991/kernelmid_dense991/biasmid_dense381/kernelmid_dense381/biasdense_6/kerneldense_6/biasmid_dense109/kernelmid_dense109/bias&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betaoutput_layer/kerneloutput_layer/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_15240814
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Î-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*÷,
valueí,Bê,VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH

SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ì
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp*input_dense2053/kernel/Read/ReadVariableOp(input_dense2053/bias/Read/ReadVariableOp'mid_dense991/kernel/Read/ReadVariableOp%mid_dense991/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp'mid_dense381/kernel/Read/ReadVariableOp%mid_dense381/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp'mid_dense109/kernel/Read/ReadVariableOp%mid_dense109/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp/RMSprop/conv2d_6/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_6/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_8/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_8/beta/rms/Read/ReadVariableOp/RMSprop/conv2d_7/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_7/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_9/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_9/beta/rms/Read/ReadVariableOp/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOp<RMSprop/batch_normalization_10/gamma/rms/Read/ReadVariableOp;RMSprop/batch_normalization_10/beta/rms/Read/ReadVariableOp6RMSprop/input_dense2053/kernel/rms/Read/ReadVariableOp4RMSprop/input_dense2053/bias/rms/Read/ReadVariableOp3RMSprop/mid_dense991/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense991/bias/rms/Read/ReadVariableOp.RMSprop/dense_6/kernel/rms/Read/ReadVariableOp,RMSprop/dense_6/bias/rms/Read/ReadVariableOp3RMSprop/mid_dense381/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense381/bias/rms/Read/ReadVariableOp<RMSprop/batch_normalization_11/gamma/rms/Read/ReadVariableOp;RMSprop/batch_normalization_11/beta/rms/Read/ReadVariableOp3RMSprop/mid_dense109/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense109/bias/rms/Read/ReadVariableOp3RMSprop/output_layer/kernel/rms/Read/ReadVariableOp1RMSprop/output_layer/bias/rms/Read/ReadVariableOp.RMSprop/dense_7/kernel/rms/Read/ReadVariableOp,RMSprop/dense_7/bias/rms/Read/ReadVariableOp.RMSprop/dense_8/kernel/rms/Read/ReadVariableOp,RMSprop/dense_8/bias/rms/Read/ReadVariableOp.RMSprop/dense_9/kernel/rms/Read/ReadVariableOp,RMSprop/dense_9/bias/rms/Read/ReadVariableOpConst"/device:CPU:0*d
dtypesZ
X2V	

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
Ñ-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*÷,
valueí,Bê,VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH

RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
À
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*î
_output_shapesÛ
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOpAssignVariableOpconv2d_6/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_1AssignVariableOpconv2d_6/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_2AssignVariableOpbatch_normalization_8/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_3AssignVariableOpbatch_normalization_8/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_4AssignVariableOp!batch_normalization_8/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_5AssignVariableOp%batch_normalization_8/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpconv2d_7/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpconv2d_7/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_8AssignVariableOpbatch_normalization_9/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_9AssignVariableOpbatch_normalization_9/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_10AssignVariableOp!batch_normalization_9/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
w
AssignVariableOp_11AssignVariableOp%batch_normalization_9/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_12AssignVariableOpconv2d_8/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_13AssignVariableOpconv2d_8/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_14AssignVariableOpbatch_normalization_10/gammaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_15AssignVariableOpbatch_normalization_10/betaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_16AssignVariableOp"batch_normalization_10/moving_meanIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_17AssignVariableOp&batch_normalization_10/moving_varianceIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_18AssignVariableOpinput_dense2053/kernelIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_19AssignVariableOpinput_dense2053/biasIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_20AssignVariableOpmid_dense991/kernelIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_21AssignVariableOpmid_dense991/biasIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_22AssignVariableOpdense_6/kernelIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_23AssignVariableOpdense_6/biasIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_24AssignVariableOpmid_dense381/kernelIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_25AssignVariableOpmid_dense381/biasIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_26AssignVariableOpbatch_normalization_11/gammaIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_27AssignVariableOpbatch_normalization_11/betaIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_28AssignVariableOp"batch_normalization_11/moving_meanIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_29AssignVariableOp&batch_normalization_11/moving_varianceIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_30AssignVariableOpmid_dense109/kernelIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_31AssignVariableOpmid_dense109/biasIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_32AssignVariableOpoutput_layer/kernelIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_33AssignVariableOpoutput_layer/biasIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_34AssignVariableOpdense_7/kernelIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_35AssignVariableOpdense_7/biasIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_36AssignVariableOpdense_8/kernelIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_37AssignVariableOpdense_8/biasIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_38AssignVariableOpdense_9/kernelIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_39AssignVariableOpdense_9/biasIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0	*
_output_shapes
:
^
AssignVariableOp_40AssignVariableOpRMSprop/iterIdentity_41"/device:CPU:0*
dtype0	
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_41AssignVariableOpRMSprop/decayIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_42AssignVariableOpRMSprop/learning_rateIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_43AssignVariableOpRMSprop/momentumIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_44AssignVariableOpRMSprop/rhoIdentity_45"/device:CPU:0*
dtype0
W
Identity_46IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_45AssignVariableOptotalIdentity_46"/device:CPU:0*
dtype0
W
Identity_47IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_46AssignVariableOpcountIdentity_47"/device:CPU:0*
dtype0
W
Identity_48IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_47AssignVariableOptotal_1Identity_48"/device:CPU:0*
dtype0
W
Identity_49IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_48AssignVariableOpcount_1Identity_49"/device:CPU:0*
dtype0
W
Identity_50IdentityRestoreV2:49"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_49AssignVariableOptrue_positivesIdentity_50"/device:CPU:0*
dtype0
W
Identity_51IdentityRestoreV2:50"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_50AssignVariableOptrue_negativesIdentity_51"/device:CPU:0*
dtype0
W
Identity_52IdentityRestoreV2:51"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_51AssignVariableOpfalse_positivesIdentity_52"/device:CPU:0*
dtype0
W
Identity_53IdentityRestoreV2:52"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_52AssignVariableOpfalse_negativesIdentity_53"/device:CPU:0*
dtype0
W
Identity_54IdentityRestoreV2:53"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_53AssignVariableOpRMSprop/conv2d_6/kernel/rmsIdentity_54"/device:CPU:0*
dtype0
W
Identity_55IdentityRestoreV2:54"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_54AssignVariableOpRMSprop/conv2d_6/bias/rmsIdentity_55"/device:CPU:0*
dtype0
W
Identity_56IdentityRestoreV2:55"/device:CPU:0*
T0*
_output_shapes
:
y
AssignVariableOp_55AssignVariableOp'RMSprop/batch_normalization_8/gamma/rmsIdentity_56"/device:CPU:0*
dtype0
W
Identity_57IdentityRestoreV2:56"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_56AssignVariableOp&RMSprop/batch_normalization_8/beta/rmsIdentity_57"/device:CPU:0*
dtype0
W
Identity_58IdentityRestoreV2:57"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_57AssignVariableOpRMSprop/conv2d_7/kernel/rmsIdentity_58"/device:CPU:0*
dtype0
W
Identity_59IdentityRestoreV2:58"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_58AssignVariableOpRMSprop/conv2d_7/bias/rmsIdentity_59"/device:CPU:0*
dtype0
W
Identity_60IdentityRestoreV2:59"/device:CPU:0*
T0*
_output_shapes
:
y
AssignVariableOp_59AssignVariableOp'RMSprop/batch_normalization_9/gamma/rmsIdentity_60"/device:CPU:0*
dtype0
W
Identity_61IdentityRestoreV2:60"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_60AssignVariableOp&RMSprop/batch_normalization_9/beta/rmsIdentity_61"/device:CPU:0*
dtype0
W
Identity_62IdentityRestoreV2:61"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_61AssignVariableOpRMSprop/conv2d_8/kernel/rmsIdentity_62"/device:CPU:0*
dtype0
W
Identity_63IdentityRestoreV2:62"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_62AssignVariableOpRMSprop/conv2d_8/bias/rmsIdentity_63"/device:CPU:0*
dtype0
W
Identity_64IdentityRestoreV2:63"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_63AssignVariableOp(RMSprop/batch_normalization_10/gamma/rmsIdentity_64"/device:CPU:0*
dtype0
W
Identity_65IdentityRestoreV2:64"/device:CPU:0*
T0*
_output_shapes
:
y
AssignVariableOp_64AssignVariableOp'RMSprop/batch_normalization_10/beta/rmsIdentity_65"/device:CPU:0*
dtype0
W
Identity_66IdentityRestoreV2:65"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_65AssignVariableOp"RMSprop/input_dense2053/kernel/rmsIdentity_66"/device:CPU:0*
dtype0
W
Identity_67IdentityRestoreV2:66"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_66AssignVariableOp RMSprop/input_dense2053/bias/rmsIdentity_67"/device:CPU:0*
dtype0
W
Identity_68IdentityRestoreV2:67"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_67AssignVariableOpRMSprop/mid_dense991/kernel/rmsIdentity_68"/device:CPU:0*
dtype0
W
Identity_69IdentityRestoreV2:68"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_68AssignVariableOpRMSprop/mid_dense991/bias/rmsIdentity_69"/device:CPU:0*
dtype0
W
Identity_70IdentityRestoreV2:69"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_69AssignVariableOpRMSprop/dense_6/kernel/rmsIdentity_70"/device:CPU:0*
dtype0
W
Identity_71IdentityRestoreV2:70"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_70AssignVariableOpRMSprop/dense_6/bias/rmsIdentity_71"/device:CPU:0*
dtype0
W
Identity_72IdentityRestoreV2:71"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_71AssignVariableOpRMSprop/mid_dense381/kernel/rmsIdentity_72"/device:CPU:0*
dtype0
W
Identity_73IdentityRestoreV2:72"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_72AssignVariableOpRMSprop/mid_dense381/bias/rmsIdentity_73"/device:CPU:0*
dtype0
W
Identity_74IdentityRestoreV2:73"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_73AssignVariableOp(RMSprop/batch_normalization_11/gamma/rmsIdentity_74"/device:CPU:0*
dtype0
W
Identity_75IdentityRestoreV2:74"/device:CPU:0*
T0*
_output_shapes
:
y
AssignVariableOp_74AssignVariableOp'RMSprop/batch_normalization_11/beta/rmsIdentity_75"/device:CPU:0*
dtype0
W
Identity_76IdentityRestoreV2:75"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_75AssignVariableOpRMSprop/mid_dense109/kernel/rmsIdentity_76"/device:CPU:0*
dtype0
W
Identity_77IdentityRestoreV2:76"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_76AssignVariableOpRMSprop/mid_dense109/bias/rmsIdentity_77"/device:CPU:0*
dtype0
W
Identity_78IdentityRestoreV2:77"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_77AssignVariableOpRMSprop/output_layer/kernel/rmsIdentity_78"/device:CPU:0*
dtype0
W
Identity_79IdentityRestoreV2:78"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_78AssignVariableOpRMSprop/output_layer/bias/rmsIdentity_79"/device:CPU:0*
dtype0
W
Identity_80IdentityRestoreV2:79"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_79AssignVariableOpRMSprop/dense_7/kernel/rmsIdentity_80"/device:CPU:0*
dtype0
W
Identity_81IdentityRestoreV2:80"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_80AssignVariableOpRMSprop/dense_7/bias/rmsIdentity_81"/device:CPU:0*
dtype0
W
Identity_82IdentityRestoreV2:81"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_81AssignVariableOpRMSprop/dense_8/kernel/rmsIdentity_82"/device:CPU:0*
dtype0
W
Identity_83IdentityRestoreV2:82"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_82AssignVariableOpRMSprop/dense_8/bias/rmsIdentity_83"/device:CPU:0*
dtype0
W
Identity_84IdentityRestoreV2:83"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_83AssignVariableOpRMSprop/dense_9/kernel/rmsIdentity_84"/device:CPU:0*
dtype0
W
Identity_85IdentityRestoreV2:84"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_84AssignVariableOpRMSprop/dense_9/bias/rmsIdentity_85"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
¢
Identity_86Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: üì.
Ê
³
__inference_loss_fn_5_15241940J
<mid_dense381_bias_regularizer_square_readvariableop_resource:m
identity¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¬
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense381_bias_regularizer_square_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%mid_dense381/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp

M
6__inference_mid_dense991_activity_regularizer_15234845
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex

M
6__inference_mid_dense109_activity_regularizer_15234981
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
+
²
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_15241632

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: V
SquareSquareRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       G
SumSum
Square:y:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿm: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
 
_user_specified_nameinputs
Á(
­
M__inference_input_dense2053_layer_call_and_return_conditional_losses_15242034

inputs2
matmul_readvariableop_resource:
û
ß.
biasadd_readvariableop_resource:	ß
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß£
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿû
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

 
_user_specified_nameinputs
Î

S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240890

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

í
__inference_loss_fn_6_15241960M
;mid_dense109_kernel_regularizer_abs_readvariableop_resource:m
identity¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense109_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ±
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense109_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense109/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp
ú
N
2__inference_max_pooling2d_6_layer_call_fn_15240913

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_15241229

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤+
³
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_15241433

inputs1
matmul_readvariableop_resource:	ým-
biasadd_readvariableop_resource:m
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: V
SquareSquareRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       G
SumSum
Square:y:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿý: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
 
_user_specified_nameinputs

í
__inference_loss_fn_8_15241991M
;output_layer_kernel_regularizer_abs_readvariableop_resource:
identity¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOpj
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;output_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ±
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;output_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)output_layer/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp
¦
3
*__inference_model_1_layer_call_fn_15238396
conv2d_6_input
input_dense2053_inputA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:L
>batch_normalization_11_assignmovingavg_readvariableop_resource:N
@batch_normalization_11_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢&batch_normalization_11/AssignMovingAvg¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢(batch_normalization_11/AssignMovingAvg_1¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp¢?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp¢?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp¢?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp¢@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp¢?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp¢Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ç
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0·
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ß
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7â
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ß
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¹
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0À
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ä
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8è
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ü
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ³
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0·
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ú
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ü
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0«
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: °
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0´
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ª
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0«
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ýz
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0ª
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¯
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0³
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0ª
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0©
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0²
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Á
#batch_normalization_11/moments/meanMeandense_6/Relu:activations:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes

:É
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:04batch_normalization_11/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_11/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_11/AssignMovingAvgAssignSubVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_11/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_11/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0©
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0²
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_2/dropout/MulMul*batch_normalization_11/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_2/dropout/ShapeShape*batch_normalization_11/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1'^batch_normalization_11/AssignMovingAvg6^batch_normalization_11/AssignMovingAvg/ReadVariableOp)^batch_normalization_11/AssignMovingAvg_18^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12P
&batch_normalization_11/AssignMovingAvg&batch_normalization_11/AssignMovingAvg2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_11/AssignMovingAvg_1(batch_normalization_11/AssignMovingAvg_12r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input

i
M__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_15241126

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241654

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê±
í&
#__inference__wrapped_model_15234561
conv2d_6_input
input_dense2053_inputI
/model_1_conv2d_6_conv2d_readvariableop_resource:>
0model_1_conv2d_6_biasadd_readvariableop_resource:C
5model_1_batch_normalization_8_readvariableop_resource:E
7model_1_batch_normalization_8_readvariableop_1_resource:T
Fmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:V
Hmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:I
/model_1_conv2d_7_conv2d_readvariableop_resource: >
0model_1_conv2d_7_biasadd_readvariableop_resource: C
5model_1_batch_normalization_9_readvariableop_resource: E
7model_1_batch_normalization_9_readvariableop_1_resource: T
Fmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: I
/model_1_conv2d_8_conv2d_readvariableop_resource: @>
0model_1_conv2d_8_biasadd_readvariableop_resource:@D
6model_1_batch_normalization_10_readvariableop_resource:@F
8model_1_batch_normalization_10_readvariableop_1_resource:@U
Gmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@W
Imodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@J
6model_1_input_dense2053_matmul_readvariableop_resource:
û
ßF
7model_1_input_dense2053_biasadd_readvariableop_resource:	ßG
3model_1_mid_dense991_matmul_readvariableop_resource:
ßýC
4model_1_mid_dense991_biasadd_readvariableop_resource:	ýF
3model_1_mid_dense381_matmul_readvariableop_resource:	ýmB
4model_1_mid_dense381_biasadd_readvariableop_resource:mA
.model_1_dense_6_matmul_readvariableop_resource:	8=
/model_1_dense_6_biasadd_readvariableop_resource:E
3model_1_mid_dense109_matmul_readvariableop_resource:mB
4model_1_mid_dense109_biasadd_readvariableop_resource:N
@model_1_batch_normalization_11_batchnorm_readvariableop_resource:R
Dmodel_1_batch_normalization_11_batchnorm_mul_readvariableop_resource:P
Bmodel_1_batch_normalization_11_batchnorm_readvariableop_1_resource:P
Bmodel_1_batch_normalization_11_batchnorm_readvariableop_2_resource:E
3model_1_output_layer_matmul_readvariableop_resource:B
4model_1_output_layer_biasadd_readvariableop_resource:@
.model_1_dense_7_matmul_readvariableop_resource:=
/model_1_dense_7_biasadd_readvariableop_resource:@
.model_1_dense_8_matmul_readvariableop_resource:=
/model_1_dense_8_biasadd_readvariableop_resource:@
.model_1_dense_9_matmul_readvariableop_resource:=
/model_1_dense_9_biasadd_readvariableop_resource:
identity¢>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢-model_1/batch_normalization_10/ReadVariableOp¢/model_1/batch_normalization_10/ReadVariableOp_1¢7model_1/batch_normalization_11/batchnorm/ReadVariableOp¢9model_1/batch_normalization_11/batchnorm/ReadVariableOp_1¢9model_1/batch_normalization_11/batchnorm/ReadVariableOp_2¢;model_1/batch_normalization_11/batchnorm/mul/ReadVariableOp¢=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢,model_1/batch_normalization_8/ReadVariableOp¢.model_1/batch_normalization_8/ReadVariableOp_1¢=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢,model_1/batch_normalization_9/ReadVariableOp¢.model_1/batch_normalization_9/ReadVariableOp_1¢'model_1/conv2d_6/BiasAdd/ReadVariableOp¢&model_1/conv2d_6/Conv2D/ReadVariableOp¢'model_1/conv2d_7/BiasAdd/ReadVariableOp¢&model_1/conv2d_7/Conv2D/ReadVariableOp¢'model_1/conv2d_8/BiasAdd/ReadVariableOp¢&model_1/conv2d_8/Conv2D/ReadVariableOp¢&model_1/dense_6/BiasAdd/ReadVariableOp¢%model_1/dense_6/MatMul/ReadVariableOp¢&model_1/dense_7/BiasAdd/ReadVariableOp¢%model_1/dense_7/MatMul/ReadVariableOp¢&model_1/dense_8/BiasAdd/ReadVariableOp¢%model_1/dense_8/MatMul/ReadVariableOp¢&model_1/dense_9/BiasAdd/ReadVariableOp¢%model_1/dense_9/MatMul/ReadVariableOp¢.model_1/input_dense2053/BiasAdd/ReadVariableOp¢-model_1/input_dense2053/MatMul/ReadVariableOp¢+model_1/mid_dense109/BiasAdd/ReadVariableOp¢*model_1/mid_dense109/MatMul/ReadVariableOp¢+model_1/mid_dense381/BiasAdd/ReadVariableOp¢*model_1/mid_dense381/MatMul/ReadVariableOp¢+model_1/mid_dense991/BiasAdd/ReadVariableOp¢*model_1/mid_dense991/MatMul/ReadVariableOp¢+model_1/output_layer/BiasAdd/ReadVariableOp¢*model_1/output_layer/MatMul/ReadVariableOp
&model_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ä
model_1/conv2d_6/Conv2DConv2Dconv2d_6_input.model_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

'model_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
model_1/conv2d_6/BiasAddBiasAdd model_1/conv2d_6/Conv2D:output:0/model_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~z
model_1/conv2d_6/ReluRelu!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
,model_1/batch_normalization_8/ReadVariableOpReadVariableOp5model_1_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0¢
.model_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0À
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ä
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0é
.model_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3#model_1/conv2d_6/Relu:activations:04model_1/batch_normalization_8/ReadVariableOp:value:06model_1/batch_normalization_8/ReadVariableOp_1:value:0Emodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
is_training( Ë
model_1/max_pooling2d_6/MaxPoolMaxPool2model_1/batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

&model_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Þ
model_1/conv2d_7/Conv2DConv2D(model_1/max_pooling2d_6/MaxPool:output:0.model_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

'model_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
model_1/conv2d_7/BiasAddBiasAdd model_1/conv2d_7/Conv2D:output:0/model_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= z
model_1/conv2d_7/ReluRelu!model_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
,model_1/batch_normalization_9/ReadVariableOpReadVariableOp5model_1_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0¢
.model_1/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0À
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ä
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0é
.model_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3#model_1/conv2d_7/Relu:activations:04model_1/batch_normalization_9/ReadVariableOp:value:06model_1/batch_normalization_9/ReadVariableOp_1:value:0Emodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
is_training( Ë
model_1/max_pooling2d_7/MaxPoolMaxPool2model_1/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

&model_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Þ
model_1/conv2d_8/Conv2DConv2D(model_1/max_pooling2d_7/MaxPool:output:0.model_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

'model_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
model_1/conv2d_8/BiasAddBiasAdd model_1/conv2d_8/Conv2D:output:0/model_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
model_1/conv2d_8/ReluRelu!model_1/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-model_1/batch_normalization_10/ReadVariableOpReadVariableOp6model_1_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_1/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0î
/model_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3#model_1/conv2d_8/Relu:activations:05model_1/batch_normalization_10/ReadVariableOp:value:07model_1/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ¦
-model_1/input_dense2053/MatMul/ReadVariableOpReadVariableOp6model_1_input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0©
model_1/input_dense2053/MatMulMatMulinput_dense2053_input5model_1/input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß£
.model_1/input_dense2053/BiasAdd/ReadVariableOpReadVariableOp7model_1_input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0¿
model_1/input_dense2053/BiasAddBiasAdd(model_1/input_dense2053/MatMul:product:06model_1/input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
model_1/input_dense2053/ReluRelu(model_1/input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
2model_1/input_dense2053/ActivityRegularizer/SquareSquare*model_1/input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
1model_1/input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ë
/model_1/input_dense2053/ActivityRegularizer/SumSum6model_1/input_dense2053/ActivityRegularizer/Square:y:0:model_1/input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: v
1model_1/input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Í
/model_1/input_dense2053/ActivityRegularizer/mulMul:model_1/input_dense2053/ActivityRegularizer/mul/x:output:08model_1/input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
1model_1/input_dense2053/ActivityRegularizer/ShapeShape*model_1/input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
?model_1/input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Amodel_1/input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Amodel_1/input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9model_1/input_dense2053/ActivityRegularizer/strided_sliceStridedSlice:model_1/input_dense2053/ActivityRegularizer/Shape:output:0Hmodel_1/input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Jmodel_1/input_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Jmodel_1/input_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¬
0model_1/input_dense2053/ActivityRegularizer/CastCastBmodel_1/input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ê
3model_1/input_dense2053/ActivityRegularizer/truedivRealDiv3model_1/input_dense2053/ActivityRegularizer/mul:z:04model_1/input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ì
model_1/max_pooling2d_8/MaxPoolMaxPool3model_1/batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
 
*model_1/mid_dense991/MatMul/ReadVariableOpReadVariableOp3model_1_mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0¸
model_1/mid_dense991/MatMulMatMul*model_1/input_dense2053/Relu:activations:02model_1/mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
+model_1/mid_dense991/BiasAdd/ReadVariableOpReadVariableOp4model_1_mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0¶
model_1/mid_dense991/BiasAddBiasAdd%model_1/mid_dense991/MatMul:product:03model_1/mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý{
model_1/mid_dense991/ReluRelu%model_1/mid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
/model_1/mid_dense991/ActivityRegularizer/SquareSquare'model_1/mid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
.model_1/mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,model_1/mid_dense991/ActivityRegularizer/SumSum3model_1/mid_dense991/ActivityRegularizer/Square:y:07model_1/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: s
.model_1/mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ä
,model_1/mid_dense991/ActivityRegularizer/mulMul7model_1/mid_dense991/ActivityRegularizer/mul/x:output:05model_1/mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
.model_1/mid_dense991/ActivityRegularizer/ShapeShape'model_1/mid_dense991/Relu:activations:0*
T0*
_output_shapes
:
<model_1/mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>model_1/mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>model_1/mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6model_1/mid_dense991/ActivityRegularizer/strided_sliceStridedSlice7model_1/mid_dense991/ActivityRegularizer/Shape:output:0Emodel_1/mid_dense991/ActivityRegularizer/strided_slice/stack:output:0Gmodel_1/mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0Gmodel_1/mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¦
-model_1/mid_dense991/ActivityRegularizer/CastCast?model_1/mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Á
0model_1/mid_dense991/ActivityRegularizer/truedivRealDiv0model_1/mid_dense991/ActivityRegularizer/mul:z:01model_1/mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: h
model_1/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   £
model_1/flatten_2/ReshapeReshape(model_1/max_pooling2d_8/MaxPool:output:0 model_1/flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
*model_1/mid_dense381/MatMul/ReadVariableOpReadVariableOp3model_1_mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0´
model_1/mid_dense381/MatMulMatMul'model_1/mid_dense991/Relu:activations:02model_1/mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+model_1/mid_dense381/BiasAdd/ReadVariableOpReadVariableOp4model_1_mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0µ
model_1/mid_dense381/BiasAddBiasAdd%model_1/mid_dense381/MatMul:product:03model_1/mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmz
model_1/mid_dense381/ReluRelu%model_1/mid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
/model_1/mid_dense381/ActivityRegularizer/SquareSquare'model_1/mid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
.model_1/mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,model_1/mid_dense381/ActivityRegularizer/SumSum3model_1/mid_dense381/ActivityRegularizer/Square:y:07model_1/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: s
.model_1/mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ä
,model_1/mid_dense381/ActivityRegularizer/mulMul7model_1/mid_dense381/ActivityRegularizer/mul/x:output:05model_1/mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
.model_1/mid_dense381/ActivityRegularizer/ShapeShape'model_1/mid_dense381/Relu:activations:0*
T0*
_output_shapes
:
<model_1/mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>model_1/mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>model_1/mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6model_1/mid_dense381/ActivityRegularizer/strided_sliceStridedSlice7model_1/mid_dense381/ActivityRegularizer/Shape:output:0Emodel_1/mid_dense381/ActivityRegularizer/strided_slice/stack:output:0Gmodel_1/mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0Gmodel_1/mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¦
-model_1/mid_dense381/ActivityRegularizer/CastCast?model_1/mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Á
0model_1/mid_dense381/ActivityRegularizer/truedivRealDiv0model_1/mid_dense381/ActivityRegularizer/mul:z:01model_1/mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0¥
model_1/dense_6/MatMulMatMul"model_1/flatten_2/Reshape:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*model_1/mid_dense109/MatMul/ReadVariableOpReadVariableOp3model_1_mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0´
model_1/mid_dense109/MatMulMatMul'model_1/mid_dense381/Relu:activations:02model_1/mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+model_1/mid_dense109/BiasAdd/ReadVariableOpReadVariableOp4model_1_mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
model_1/mid_dense109/BiasAddBiasAdd%model_1/mid_dense109/MatMul:product:03model_1/mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model_1/mid_dense109/ReluRelu%model_1/mid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/model_1/mid_dense109/ActivityRegularizer/SquareSquare'model_1/mid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.model_1/mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,model_1/mid_dense109/ActivityRegularizer/SumSum3model_1/mid_dense109/ActivityRegularizer/Square:y:07model_1/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: s
.model_1/mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ä
,model_1/mid_dense109/ActivityRegularizer/mulMul7model_1/mid_dense109/ActivityRegularizer/mul/x:output:05model_1/mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
.model_1/mid_dense109/ActivityRegularizer/ShapeShape'model_1/mid_dense109/Relu:activations:0*
T0*
_output_shapes
:
<model_1/mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>model_1/mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>model_1/mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6model_1/mid_dense109/ActivityRegularizer/strided_sliceStridedSlice7model_1/mid_dense109/ActivityRegularizer/Shape:output:0Emodel_1/mid_dense109/ActivityRegularizer/strided_slice/stack:output:0Gmodel_1/mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0Gmodel_1/mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¦
-model_1/mid_dense109/ActivityRegularizer/CastCast?model_1/mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Á
0model_1/mid_dense109/ActivityRegularizer/truedivRealDiv0model_1/mid_dense109/ActivityRegularizer/mul:z:01model_1/mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ´
7model_1/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp@model_1_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_1/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ô
,model_1/batch_normalization_11/batchnorm/addAddV2?model_1/batch_normalization_11/batchnorm/ReadVariableOp:value:07model_1/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:
.model_1/batch_normalization_11/batchnorm/RsqrtRsqrt0model_1/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¼
;model_1/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_1_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ñ
,model_1/batch_normalization_11/batchnorm/mulMul2model_1/batch_normalization_11/batchnorm/Rsqrt:y:0Cmodel_1/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:½
.model_1/batch_normalization_11/batchnorm/mul_1Mul"model_1/dense_6/Relu:activations:00model_1/batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
9model_1/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_1_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ï
.model_1/batch_normalization_11/batchnorm/mul_2MulAmodel_1/batch_normalization_11/batchnorm/ReadVariableOp_1:value:00model_1/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¸
9model_1/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_1_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ï
,model_1/batch_normalization_11/batchnorm/subSubAmodel_1/batch_normalization_11/batchnorm/ReadVariableOp_2:value:02model_1/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ï
.model_1/batch_normalization_11/batchnorm/add_1AddV22model_1/batch_normalization_11/batchnorm/mul_1:z:00model_1/batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*model_1/output_layer/MatMul/ReadVariableOpReadVariableOp3model_1_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0´
model_1/output_layer/MatMulMatMul'model_1/mid_dense109/Relu:activations:02model_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+model_1/output_layer/BiasAdd/ReadVariableOpReadVariableOp4model_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
model_1/output_layer/BiasAddBiasAdd%model_1/output_layer/MatMul:product:03model_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/output_layer/SigmoidSigmoid%model_1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/model_1/output_layer/ActivityRegularizer/SquareSquare model_1/output_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.model_1/output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,model_1/output_layer/ActivityRegularizer/SumSum3model_1/output_layer/ActivityRegularizer/Square:y:07model_1/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: s
.model_1/output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ä
,model_1/output_layer/ActivityRegularizer/mulMul7model_1/output_layer/ActivityRegularizer/mul/x:output:05model_1/output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ~
.model_1/output_layer/ActivityRegularizer/ShapeShape model_1/output_layer/Sigmoid:y:0*
T0*
_output_shapes
:
<model_1/output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>model_1/output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>model_1/output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6model_1/output_layer/ActivityRegularizer/strided_sliceStridedSlice7model_1/output_layer/ActivityRegularizer/Shape:output:0Emodel_1/output_layer/ActivityRegularizer/strided_slice/stack:output:0Gmodel_1/output_layer/ActivityRegularizer/strided_slice/stack_1:output:0Gmodel_1/output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¦
-model_1/output_layer/ActivityRegularizer/CastCast?model_1/output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Á
0model_1/output_layer/ActivityRegularizer/truedivRealDiv0model_1/output_layer/ActivityRegularizer/mul:z:01model_1/output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
model_1/dropout_2/IdentityIdentity2model_1/batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¦
model_1/dense_7/MatMulMatMul#model_1/dropout_2/Identity:output:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model_1/dense_7/SoftmaxSoftmax model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
model_1/activation_4/SoftmaxSoftmax model_1/output_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
model_1/concatenate_1/concatConcatV2!model_1/dense_7/Softmax:softmax:0&model_1/activation_4/Softmax:softmax:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
model_1/dense_8/MatMulMatMul%model_1/concatenate_1/concat:output:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model_1/dense_8/ReluRelu model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¥
model_1/dense_9/MatMulMatMul"model_1/dense_8/Relu:activations:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model_1/dense_9/SoftmaxSoftmax model_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!model_1/dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp?^model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_10/ReadVariableOp0^model_1/batch_normalization_10/ReadVariableOp_18^model_1/batch_normalization_11/batchnorm/ReadVariableOp:^model_1/batch_normalization_11/batchnorm/ReadVariableOp_1:^model_1/batch_normalization_11/batchnorm/ReadVariableOp_2<^model_1/batch_normalization_11/batchnorm/mul/ReadVariableOp>^model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_8/ReadVariableOp/^model_1/batch_normalization_8/ReadVariableOp_1>^model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_9/ReadVariableOp/^model_1/batch_normalization_9/ReadVariableOp_1(^model_1/conv2d_6/BiasAdd/ReadVariableOp'^model_1/conv2d_6/Conv2D/ReadVariableOp(^model_1/conv2d_7/BiasAdd/ReadVariableOp'^model_1/conv2d_7/Conv2D/ReadVariableOp(^model_1/conv2d_8/BiasAdd/ReadVariableOp'^model_1/conv2d_8/Conv2D/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp&^model_1/dense_9/MatMul/ReadVariableOp/^model_1/input_dense2053/BiasAdd/ReadVariableOp.^model_1/input_dense2053/MatMul/ReadVariableOp,^model_1/mid_dense109/BiasAdd/ReadVariableOp+^model_1/mid_dense109/MatMul/ReadVariableOp,^model_1/mid_dense381/BiasAdd/ReadVariableOp+^model_1/mid_dense381/MatMul/ReadVariableOp,^model_1/mid_dense991/BiasAdd/ReadVariableOp+^model_1/mid_dense991/MatMul/ReadVariableOp,^model_1/output_layer/BiasAdd/ReadVariableOp+^model_1/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_10/ReadVariableOp-model_1/batch_normalization_10/ReadVariableOp2b
/model_1/batch_normalization_10/ReadVariableOp_1/model_1/batch_normalization_10/ReadVariableOp_12r
7model_1/batch_normalization_11/batchnorm/ReadVariableOp7model_1/batch_normalization_11/batchnorm/ReadVariableOp2v
9model_1/batch_normalization_11/batchnorm/ReadVariableOp_19model_1/batch_normalization_11/batchnorm/ReadVariableOp_12v
9model_1/batch_normalization_11/batchnorm/ReadVariableOp_29model_1/batch_normalization_11/batchnorm/ReadVariableOp_22z
;model_1/batch_normalization_11/batchnorm/mul/ReadVariableOp;model_1/batch_normalization_11/batchnorm/mul/ReadVariableOp2~
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_8/ReadVariableOp,model_1/batch_normalization_8/ReadVariableOp2`
.model_1/batch_normalization_8/ReadVariableOp_1.model_1/batch_normalization_8/ReadVariableOp_12~
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_9/ReadVariableOp,model_1/batch_normalization_9/ReadVariableOp2`
.model_1/batch_normalization_9/ReadVariableOp_1.model_1/batch_normalization_9/ReadVariableOp_12R
'model_1/conv2d_6/BiasAdd/ReadVariableOp'model_1/conv2d_6/BiasAdd/ReadVariableOp2P
&model_1/conv2d_6/Conv2D/ReadVariableOp&model_1/conv2d_6/Conv2D/ReadVariableOp2R
'model_1/conv2d_7/BiasAdd/ReadVariableOp'model_1/conv2d_7/BiasAdd/ReadVariableOp2P
&model_1/conv2d_7/Conv2D/ReadVariableOp&model_1/conv2d_7/Conv2D/ReadVariableOp2R
'model_1/conv2d_8/BiasAdd/ReadVariableOp'model_1/conv2d_8/BiasAdd/ReadVariableOp2P
&model_1/conv2d_8/Conv2D/ReadVariableOp&model_1/conv2d_8/Conv2D/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2N
%model_1/dense_9/MatMul/ReadVariableOp%model_1/dense_9/MatMul/ReadVariableOp2`
.model_1/input_dense2053/BiasAdd/ReadVariableOp.model_1/input_dense2053/BiasAdd/ReadVariableOp2^
-model_1/input_dense2053/MatMul/ReadVariableOp-model_1/input_dense2053/MatMul/ReadVariableOp2Z
+model_1/mid_dense109/BiasAdd/ReadVariableOp+model_1/mid_dense109/BiasAdd/ReadVariableOp2X
*model_1/mid_dense109/MatMul/ReadVariableOp*model_1/mid_dense109/MatMul/ReadVariableOp2Z
+model_1/mid_dense381/BiasAdd/ReadVariableOp+model_1/mid_dense381/BiasAdd/ReadVariableOp2X
*model_1/mid_dense381/MatMul/ReadVariableOp*model_1/mid_dense381/MatMul/ReadVariableOp2Z
+model_1/mid_dense991/BiasAdd/ReadVariableOp+model_1/mid_dense991/BiasAdd/ReadVariableOp2X
*model_1/mid_dense991/MatMul/ReadVariableOp*model_1/mid_dense991/MatMul/ReadVariableOp2Z
+model_1/output_layer/BiasAdd/ReadVariableOp+model_1/output_layer/BiasAdd/ReadVariableOp2X
*model_1/output_layer/MatMul/ReadVariableOp*model_1/output_layer/MatMul/ReadVariableOp:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input
Ú	
K
,__inference_dropout_2_layer_call_fn_15241649

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Û
*__inference_dense_7_layer_call_fn_15241768

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦(

2__inference_input_dense2053_layer_call_fn_15241179

inputs2
matmul_readvariableop_resource:
û
ß.
biasadd_readvariableop_resource:	ß
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß£
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿû
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

 
_user_specified_nameinputs
Ó
f
J__inference_activation_4_layer_call_and_return_conditional_losses_15241789

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

M
6__inference_mid_dense381_activity_regularizer_15234858
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex

P
9__inference_input_dense2053_activity_regularizer_15234832
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
õ	
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241666

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú$
Ò
9__inference_batch_normalization_11_layer_call_fn_15241487

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
§
8__inference_batch_normalization_9_layer_call_fn_15240976

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

ä
+__inference_conv2d_7_layer_call_fn_15240929

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ)?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?
 
_user_specified_nameinputs


ö
E__inference_dense_8_layer_call_and_return_conditional_losses_15241825

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
õ3
E__inference_model_1_layer_call_and_return_conditional_losses_15239279
conv2d_6_input
input_dense2053_inputA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:L
>batch_normalization_11_assignmovingavg_readvariableop_resource:N
@batch_normalization_11_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢&batch_normalization_11/AssignMovingAvg¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢(batch_normalization_11/AssignMovingAvg_1¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp¢?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp¢?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp¢?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp¢@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp¢?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp¢Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ç
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0·
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ß
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7â
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ß
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¹
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0À
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ä
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8è
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ü
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ³
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0·
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ú
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ü
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0«
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: °
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0´
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ª
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0«
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ýz
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0ª
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¯
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0³
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0ª
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0©
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0²
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Á
#batch_normalization_11/moments/meanMeandense_6/Relu:activations:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes

:É
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:04batch_normalization_11/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_11/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_11/AssignMovingAvgAssignSubVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_11/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_11/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0©
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0²
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_2/dropout/MulMul*batch_normalization_11/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_2/dropout/ShapeShape*batch_normalization_11/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_1Identity/input_dense2053/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,mid_dense991/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_3Identity,mid_dense381/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_4Identity,mid_dense109/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_5Identity,output_layer/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ¶
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1'^batch_normalization_11/AssignMovingAvg6^batch_normalization_11/AssignMovingAvg/ReadVariableOp)^batch_normalization_11/AssignMovingAvg_18^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12P
&batch_normalization_11/AssignMovingAvg&batch_normalization_11/AssignMovingAvg2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_11/AssignMovingAvg_1(batch_normalization_11/AssignMovingAvg_12r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input
ë

ä
+__inference_conv2d_8_layer_call_fn_15241033

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ú
N
2__inference_max_pooling2d_8_layer_call_fn_15241121

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ö
E__inference_dense_9_layer_call_and_return_conditional_losses_15241847

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ø
__inference_loss_fn_0_15241867R
>input_dense2053_kernel_regularizer_abs_readvariableop_resource:
û
ß
identity¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOpm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>input_dense2053_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¹
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>input_dense2053_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentity,input_dense2053/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ¹
NoOpNoOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp
®
H
,__inference_flatten_2_layer_call_fn_15241223

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í

ä
+__inference_conv2d_6_layer_call_fn_15240825

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿU: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15240994

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_15240918

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_15241022

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò,
Á
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_15241217

inputs2
matmul_readvariableop_resource:
û
ß.
biasadd_readvariableop_resource:	ß
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: W
SquareSquareRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       G
SumSum
Square:y:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: £
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿû
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

 
_user_specified_nameinputs
ù&

/__inference_mid_dense381_layer_call_fn_15241395

inputs1
matmul_readvariableop_resource:	ým-
biasadd_readvariableop_resource:m
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿý: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
 
_user_specified_nameinputs
'

J__inference_mid_dense381_layer_call_and_return_conditional_losses_15242098

inputs1
matmul_readvariableop_resource:	ým-
biasadd_readvariableop_resource:m
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿý: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
 
_user_specified_nameinputs

M
6__inference_output_layer_activity_regularizer_15234994
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ú
N
2__inference_max_pooling2d_7_layer_call_fn_15241017

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

º
__inference_loss_fn_1_15241878N
?input_dense2053_bias_regularizer_square_readvariableop_resource:	ß
identity¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp³
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp?input_dense2053_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentity(input_dense2053/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp
ñ×
ó*
*__inference_model_1_layer_call_fn_15240053
inputs_0
inputs_1A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:L
>batch_normalization_11_assignmovingavg_readvariableop_resource:N
@batch_normalization_11_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢&batch_normalization_11/AssignMovingAvg¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢(batch_normalization_11/AssignMovingAvg_1¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d_6/Conv2DConv2Dinputs_0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ç
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinputs_1-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Á
#batch_normalization_11/moments/meanMeandense_6/Relu:activations:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes

:É
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:04batch_normalization_11/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_11/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_11/AssignMovingAvgAssignSubVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_11/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_11/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_2/dropout/MulMul*batch_normalization_11/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_2/dropout/ShapeShape*batch_normalization_11/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1'^batch_normalization_11/AssignMovingAvg6^batch_normalization_11/AssignMovingAvg/ReadVariableOp)^batch_normalization_11/AssignMovingAvg_18^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12P
&batch_normalization_11/AssignMovingAvg&batch_normalization_11/AssignMovingAvg2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_11/AssignMovingAvg_1(batch_normalization_11/AssignMovingAvg_12r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

"
_user_specified_name
inputs/1
'

/__inference_mid_dense991_layer_call_fn_15241282

inputs2
matmul_readvariableop_resource:
ßý.
biasadd_readvariableop_resource:	ý
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýj
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿß: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
 
_user_specified_nameinputs
³

8__inference_batch_normalization_9_layer_call_fn_15240958

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 

÷
E__inference_dense_6_layer_call_and_return_conditional_losses_15241342

inputs1
matmul_readvariableop_resource:	8-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15241012

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡

ö
E__inference_dense_7_layer_call_and_return_conditional_losses_15241779

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

9__inference_batch_normalization_10_layer_call_fn_15241062

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³+
µ
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_15241320

inputs2
matmul_readvariableop_resource:
ßý.
biasadd_readvariableop_resource:	ý
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýj
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: W
SquareSquareRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       G
SumSum
Square:y:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿß: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
 
_user_specified_nameinputs
¶

9__inference_batch_normalization_11_layer_call_fn_15241453

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýÐ
°0
*__inference_model_1_layer_call_fn_15235421
conv2d_6_input
input_dense2053_inputA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:H
:batch_normalization_11_batchnorm_readvariableop_1_resource:H
:batch_normalization_11_batchnorm_readvariableop_2_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp¢?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp¢?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp¢?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp¢@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp¢?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp¢Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¹
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
is_training( »
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¹
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
is_training( »
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¾
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0·
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ß
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7â
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ß
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¹
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0À
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ä
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8è
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ü
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ³
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0·
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ú
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ü
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0«
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: °
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0´
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ª
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0«
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ýz
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0ª
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¯
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0³
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0ª
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0©
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0²
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0©
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0²
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: |
dropout_2/IdentityIdentity*batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_10^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input
ñ&

/__inference_mid_dense109_layer_call_fn_15241594

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿm: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
 
_user_specified_nameinputs
Ê
³
__inference_loss_fn_7_15241971J
<mid_dense109_bias_regularizer_square_readvariableop_resource:
identity¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¬
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense109_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%mid_dense109/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp
À×
1
E__inference_model_1_layer_call_and_return_conditional_losses_15238827
conv2d_6_input
input_dense2053_inputA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:H
:batch_normalization_11_batchnorm_readvariableop_1_resource:H
:batch_normalization_11_batchnorm_readvariableop_2_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp¢Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp¢?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp¢?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp¢?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp¢@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp¢?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp¢Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¹
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
is_training( »
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¹
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
is_training( »
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¾
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¶
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0·
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ß
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7â
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ß
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¹
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0À
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ä
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8è
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ü
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ³
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0·
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ú
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ü
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0«
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: °
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0´
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßý
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ª
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0«
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ýz
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0ª
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¯
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0³
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ým
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0ª
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0©
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0²
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0©
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       Í
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Ð
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Í
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ®
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0²
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       Ò
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ö
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: Ê
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ©
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ª
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8Ê
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: |
dropout_2/IdentityIdentity*batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_1Identity/input_dense2053/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,mid_dense991/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_3Identity,mid_dense381/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_4Identity,mid_dense109/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_5Identity,output_layer/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: æ
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_10^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input
¸
K
/__inference_activation_4_layer_call_fn_15241784

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¨
9__inference_batch_normalization_10_layer_call_fn_15241080

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240908

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
)
E__inference_model_1_layer_call_and_return_conditional_losses_15240379
inputs_0
inputs_1A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:H
:batch_normalization_11_batchnorm_readvariableop_1_resource:H
:batch_normalization_11_batchnorm_readvariableop_2_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d_6/Conv2DConv2Dinputs_0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¹
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
is_training( »
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¹
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
is_training( »
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¾
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinputs_1-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: |
dropout_2/IdentityIdentity*batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_1Identity/input_dense2053/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,mid_dense991/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_3Identity,mid_dense381/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_4Identity,mid_dense109/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_5Identity,output_layer/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: â
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_10^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

"
_user_specified_name
inputs/1
¿
J
,__inference_dropout_2_layer_call_fn_15241637

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_8_layer_call_and_return_conditional_losses_15241044

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
é
Õ	
&__inference_signature_wrapper_15240814
conv2d_6_input
input_dense2053_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
û
ß

unknown_18:	ß

unknown_19:
ßý

unknown_20:	ý

unknown_21:	ým

unknown_22:m

unknown_23:	8

unknown_24:

unknown_25:m

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputinput_dense2053_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_15234561o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
(
_user_specified_nameconv2d_6_input:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

/
_user_specified_nameinput_dense2053_input


Ü
*__inference_dense_6_layer_call_fn_15241331

inputs1
matmul_readvariableop_resource:	8-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241116

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢'
¡
J__inference_mid_dense991_layer_call_and_return_conditional_losses_15242066

inputs2
matmul_readvariableop_resource:
ßý.
biasadd_readvariableop_resource:	ý
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýj
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿß: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
 
_user_specified_nameinputs
ð&

/__inference_output_layer_layer_call_fn_15241719

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_7_layer_call_and_return_conditional_losses_15240940

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ)?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?
 
_user_specified_nameinputs


Û
*__inference_dense_8_layer_call_fn_15241814

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
´
__inference_loss_fn_3_15241909K
<mid_dense991_bias_regularizer_square_readvariableop_resource:	ý
identity¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp­
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense991_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%mid_dense991/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp
Ñ
³
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241507

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
+
²
N__inference_output_layer_layer_call_and_return_all_conditional_losses_15241757

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: O
SquareSquareSigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       G
SumSum
Square:y:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241098

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
'

J__inference_mid_dense109_layer_call_and_return_conditional_losses_15242130

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿm: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
 
_user_specified_nameinputs
´Þ
Þ+
E__inference_model_1_layer_call_and_return_conditional_losses_15240726
inputs_0
inputs_1A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:L
>batch_normalization_11_assignmovingavg_readvariableop_resource:N
@batch_normalization_11_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢&batch_normalization_11/AssignMovingAvg¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢(batch_normalization_11/AssignMovingAvg_1¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d_6/Conv2DConv2Dinputs_0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ç
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0»
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinputs_1-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Á
#batch_normalization_11/moments/meanMeandense_6/Relu:activations:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes

:É
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:04batch_normalization_11/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_11/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_11/AssignMovingAvgAssignSubVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_11/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_11/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_2/dropout/MulMul*batch_normalization_11/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_2/dropout/ShapeShape*batch_normalization_11/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_1Identity/input_dense2053/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,mid_dense991/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_3Identity,mid_dense381/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_4Identity,mid_dense109/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: l

Identity_5Identity,output_layer/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ²
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1'^batch_normalization_11/AssignMovingAvg6^batch_normalization_11/AssignMovingAvg/ReadVariableOp)^batch_normalization_11/AssignMovingAvg_18^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12P
&batch_normalization_11/AssignMovingAvg&batch_normalization_11/AssignMovingAvg2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_11/AssignMovingAvg_1(batch_normalization_11/AssignMovingAvg_12r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

"
_user_specified_name
inputs/1
Á
§
8__inference_batch_normalization_8_layer_call_fn_15240872

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Û
*__inference_dense_9_layer_call_fn_15241836

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_6_layer_call_and_return_conditional_losses_15240836

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿU: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
 
_user_specified_nameinputs
'

J__inference_output_layer_layer_call_and_return_conditional_losses_15242162

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
³
__inference_loss_fn_9_15242002J
<output_layer_bias_regularizer_square_readvariableop_resource:
identity¢3output_layer/bias/Regularizer/Square/ReadVariableOp¬
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp<output_layer_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%output_layer/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^output_layer/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp
Ä
w
K__inference_concatenate_1_layer_call_and_return_conditional_losses_15241803
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
©
\
0__inference_concatenate_1_layer_call_fn_15241796
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
%
í
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241541

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

8__inference_batch_normalization_8_layer_call_fn_15240854

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
ï
__inference_loss_fn_2_15241898O
;mid_dense991_kernel_regularizer_abs_readvariableop_resource:
ßý
identity¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense991_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ³
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense991_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense991/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp
È¡
(
*__inference_model_1_layer_call_fn_15239711
inputs_0
inputs_1A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_8_conv2d_readvariableop_resource: @6
(conv2d_8_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
.input_dense2053_matmul_readvariableop_resource:
û
ß>
/input_dense2053_biasadd_readvariableop_resource:	ß?
+mid_dense991_matmul_readvariableop_resource:
ßý;
,mid_dense991_biasadd_readvariableop_resource:	ý>
+mid_dense381_matmul_readvariableop_resource:	ým:
,mid_dense381_biasadd_readvariableop_resource:m9
&dense_6_matmul_readvariableop_resource:	85
'dense_6_biasadd_readvariableop_resource:=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:F
8batch_normalization_11_batchnorm_readvariableop_resource:J
<batch_normalization_11_batchnorm_mul_readvariableop_resource:H
:batch_normalization_11_batchnorm_readvariableop_1_resource:H
:batch_normalization_11_batchnorm_readvariableop_2_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢5batch_normalization_9/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_9/ReadVariableOp¢&batch_normalization_9/ReadVariableOp_1¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp¢conv2d_7/BiasAdd/ReadVariableOp¢conv2d_7/Conv2D/ReadVariableOp¢conv2d_8/BiasAdd/ReadVariableOp¢conv2d_8/Conv2D/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢&input_dense2053/BiasAdd/ReadVariableOp¢%input_dense2053/MatMul/ReadVariableOp¢6input_dense2053/bias/Regularizer/Square/ReadVariableOp¢5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp¢8input_dense2053/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense109/BiasAdd/ReadVariableOp¢"mid_dense109/MatMul/ReadVariableOp¢3mid_dense109/bias/Regularizer/Square/ReadVariableOp¢2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense109/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense381/BiasAdd/ReadVariableOp¢"mid_dense381/MatMul/ReadVariableOp¢3mid_dense381/bias/Regularizer/Square/ReadVariableOp¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOp¢#mid_dense991/BiasAdd/ReadVariableOp¢"mid_dense991/MatMul/ReadVariableOp¢3mid_dense991/bias/Regularizer/Square/ReadVariableOp¢2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense991/kernel/Regularizer/Square/ReadVariableOp¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢3output_layer/bias/Regularizer/Square/ReadVariableOp¢2output_layer/kernel/Regularizer/Abs/ReadVariableOp¢5output_layer/kernel/Regularizer/Square/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d_6/Conv2DConv2Dinputs_0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿS~
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¹
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿS~:::::*
epsilon%o:*
is_training( »
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)?*
ksize
*
paddingVALID*
strides

conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Æ
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= *
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'= 
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¹
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ'= : : : : :*
epsilon%o:*
is_training( »
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Æ
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¾
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_8/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
input_dense2053/MatMulMatMulinputs_1-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0§
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßq
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿßz
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7µ
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ²
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¼
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0 
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýk
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿýw
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense991/ActivityRegularizer/mulMul/mid_dense991/ActivityRegularizer/mul/x:output:0-mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense991/ActivityRegularizer/ShapeShapemid_dense991/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿmw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense381/ActivityRegularizer/mulMul/mid_dense381/ActivityRegularizer/mul/x:output:0-mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense381/ActivityRegularizer/ShapeShapemid_dense381/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	8*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$mid_dense109/ActivityRegularizer/mulMul/mid_dense109/ActivityRegularizer/mul/x:output:0-mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: u
&mid_dense109/ActivityRegularizer/ShapeShapemid_dense109/Relu:activations:0*
T0*
_output_shapes
:~
4mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¥
&batch_normalization_11/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ª
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¬
$output_layer/ActivityRegularizer/mulMul/output_layer/ActivityRegularizer/mul/x:output:0-output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&output_layer/ActivityRegularizer/ShapeShapeoutput_layer/Sigmoid:y:0*
T0*
_output_shapes
:~
4output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ©
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: |
dropout_2/IdentityIdentity*batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_7/MatMulMatMuldropout_2/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_4/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :º
concatenate_1/concatConcatV2dense_7/Softmax:softmax:0activation_4/Softmax:softmax:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_8/MatMulMatMulconcatenate_1/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¯
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7²
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¯
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ©
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
û
ß*
dtype0 
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
û
ß{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ´
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¸
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ¬
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: £
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:ß*
dtype0
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ßp
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¬
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
ßý*
dtype0
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ßýx
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:ý*
dtype0
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:ým
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¢
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¡
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_10^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿU:ÿÿÿÿÿÿÿÿÿû
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû

"
_user_specified_name
inputs/1

î
__inference_loss_fn_4_15241929N
;mid_dense381_kernel_regularizer_abs_readvariableop_resource:	ým
identity¢2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp¢5mid_dense381/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense381_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	ým*
dtype0
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¦
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7©
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¦
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ²
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense381_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	ým*
dtype0
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ýmx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       «
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8¯
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: £
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense381/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp"Û-
saver_filename:0
Identity:0Identity_868"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
R
conv2d_6_input@
 serving_default_conv2d_6_input:0ÿÿÿÿÿÿÿÿÿU
X
input_dense2053_input?
'serving_default_input_dense2053_input:0ÿÿÿÿÿÿÿÿÿû
;
dense_90
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ë
Ý
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"
signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
¥
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
»

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
¿

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	 bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«_random_generator
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
®kernel
	¯bias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¶kernel
	·bias
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Êkernel
	Ëbias
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Òkernel
	Óbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
ç
	Úiter

Ûdecay
Ülearning_rate
Ýmomentum
Þrho
#rmsþ
$rmsÿ
,rms
-rms
<rms
=rms
Erms
Frms
Urms
Vrms
^rms
_rms
nrms
orms
|rms
}rmsrmsrmsrmsrmsrmsrmsrms rms®rms¯rms¶rms·rmsÊrmsËrmsÒrmsÓrms"
	optimizer
è
#0
$1
,2
-3
.4
/5
<6
=7
E8
F9
G10
H11
U12
V13
^14
_15
`16
a17
n18
o19
|20
}21
22
23
24
25
26
27
28
29
30
 31
®32
¯33
¶34
·35
Ê36
Ë37
Ò38
Ó39"
trackable_list_wrapper
¦
#0
$1
,2
-3
<4
=5
E6
F7
U8
V9
^10
_11
n12
o13
|14
}15
16
17
18
19
20
21
22
 23
®24
¯25
¶26
·27
Ê28
Ë29
Ò30
Ó31"
trackable_list_wrapper
p
ß0
à1
á2
â3
ã4
ä5
å6
æ7
ç8
è9"
trackable_list_wrapper
Ï
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_model_1_layer_call_fn_15235421
*__inference_model_1_layer_call_fn_15239711
*__inference_model_1_layer_call_fn_15240053
*__inference_model_1_layer_call_fn_15238396À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_model_1_layer_call_and_return_conditional_losses_15240379
E__inference_model_1_layer_call_and_return_conditional_losses_15240726
E__inference_model_1_layer_call_and_return_conditional_losses_15238827
E__inference_model_1_layer_call_and_return_conditional_losses_15239279À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ìBé
#__inference__wrapped_model_15234561conv2d_6_inputinput_dense2053_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
îserving_default"
signature_map
):'2conv2d_6/kernel
:2conv2d_6/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_6_layer_call_fn_15240825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_6_layer_call_and_return_conditional_losses_15240836¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_8_layer_call_fn_15240854
8__inference_batch_normalization_8_layer_call_fn_15240872´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240890
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240908´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_6_layer_call_fn_15240913¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_15240918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):' 2conv2d_7/kernel
: 2conv2d_7/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_7_layer_call_fn_15240929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_7_layer_call_and_return_conditional_losses_15240940¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_9_layer_call_fn_15240958
8__inference_batch_normalization_9_layer_call_fn_15240976´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15240994
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15241012´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_7_layer_call_fn_15241017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_15241022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):' @2conv2d_8/kernel
:@2conv2d_8/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_8_layer_call_fn_15241033¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_8_layer_call_and_return_conditional_losses_15241044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
<
^0
_1
`2
a3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_10_layer_call_fn_15241062
9__inference_batch_normalization_10_layer_call_fn_15241080´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241098
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241116´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_8_layer_call_fn_15241121¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_15241126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(
û
ß2input_dense2053/kernel
#:!ß2input_dense2053/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
Ñ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
¡activity_regularizer_fn
*u&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_input_dense2053_layer_call_fn_15241179¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_15241217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_2_layer_call_fn_15241223¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_flatten_2_layer_call_and_return_conditional_losses_15241229¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
ßý2mid_dense991/kernel
 :ý2mid_dense991/bias
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
Ô
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
­activity_regularizer_fn
+&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_mid_dense991_layer_call_fn_15241282¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_15241320¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:	82dense_6/kernel
:2dense_6/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_6_layer_call_fn_15241331¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_6_layer_call_and_return_conditional_losses_15241342¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$	ým2mid_dense381/kernel
:m2mid_dense381/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
ã0
ä1"
trackable_list_wrapper
Ö
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
	variables
trainable_variables
regularization_losses
__call__
¹activity_regularizer_fn
+&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_mid_dense381_layer_call_fn_15241395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_15241433¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_11_layer_call_fn_15241453
9__inference_batch_normalization_11_layer_call_fn_15241487´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241507
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241541´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
%:#m2mid_dense109/kernel
:2mid_dense109/bias
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
Ö
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
Åactivity_regularizer_fn
+¦&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_mid_dense109_layer_call_fn_15241594¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_15241632¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
,__inference_dropout_2_layer_call_fn_15241637
,__inference_dropout_2_layer_call_fn_15241649´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241654
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241666´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
%:#2output_layer/kernel
:2output_layer/bias
0
®0
¯1"
trackable_list_wrapper
0
®0
¯1"
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
Ö
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
Ñactivity_regularizer_fn
+µ&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_output_layer_layer_call_fn_15241719¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_output_layer_layer_call_and_return_all_conditional_losses_15241757¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_7/kernel
:2dense_7/bias
0
¶0
·1"
trackable_list_wrapper
0
¶0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_7_layer_call_fn_15241768¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_7_layer_call_and_return_conditional_losses_15241779¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_activation_4_layer_call_fn_15241784¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_activation_4_layer_call_and_return_conditional_losses_15241789¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_concatenate_1_layer_call_fn_15241796¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_concatenate_1_layer_call_and_return_conditional_losses_15241803¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_8/kernel
:2dense_8/bias
0
Ê0
Ë1"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_8_layer_call_fn_15241814¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_8_layer_call_and_return_conditional_losses_15241825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_9/kernel
:2dense_9/bias
0
Ò0
Ó1"
trackable_list_wrapper
0
Ò0
Ó1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_9_layer_call_fn_15241836¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_9_layer_call_and_return_conditional_losses_15241847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
µ2²
__inference_loss_fn_0_15241867
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_1_15241878
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_2_15241898
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_3_15241909
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_4_15241929
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_5_15241940
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_6_15241960
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_7_15241971
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_8_15241991
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_9_15242002
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
Z
.0
/1
G2
H3
`4
a5
6
7"
trackable_list_wrapper
Þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
8
ì0
í1
î2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
éBæ
&__inference_signature_wrapper_15240814conv2d_6_inputinput_dense2053_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ê2ç
9__inference_input_dense2053_activity_regularizer_15234832©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
÷2ô
M__inference_input_dense2053_layer_call_and_return_conditional_losses_15242034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ç2ä
6__inference_mid_dense991_activity_regularizer_15234845©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ô2ñ
J__inference_mid_dense991_layer_call_and_return_conditional_losses_15242066¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ã0
ä1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ç2ä
6__inference_mid_dense381_activity_regularizer_15234858©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ô2ñ
J__inference_mid_dense381_layer_call_and_return_conditional_losses_15242098¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ç2ä
6__inference_mid_dense109_activity_regularizer_15234981©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ô2ñ
J__inference_mid_dense109_layer_call_and_return_conditional_losses_15242130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ç2ä
6__inference_output_layer_activity_regularizer_15234994©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ô2ñ
J__inference_output_layer_layer_call_and_return_conditional_losses_15242162¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

ïtotal

ðcount
ñ	variables
ò	keras_api"
_tf_keras_metric
c

ótotal

ôcount
õ
_fn_kwargs
ö	variables
÷	keras_api"
_tf_keras_metric

øtrue_positives
ùtrue_negatives
úfalse_positives
ûfalse_negatives
ü	variables
ý	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ï0
ð1"
trackable_list_wrapper
.
ñ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ó0
ô1"
trackable_list_wrapper
.
ö	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
@
ø0
ù1
ú2
û3"
trackable_list_wrapper
.
ü	variables"
_generic_user_object
3:12RMSprop/conv2d_6/kernel/rms
%:#2RMSprop/conv2d_6/bias/rms
3:12'RMSprop/batch_normalization_8/gamma/rms
2:02&RMSprop/batch_normalization_8/beta/rms
3:1 2RMSprop/conv2d_7/kernel/rms
%:# 2RMSprop/conv2d_7/bias/rms
3:1 2'RMSprop/batch_normalization_9/gamma/rms
2:0 2&RMSprop/batch_normalization_9/beta/rms
3:1 @2RMSprop/conv2d_8/kernel/rms
%:#@2RMSprop/conv2d_8/bias/rms
4:2@2(RMSprop/batch_normalization_10/gamma/rms
3:1@2'RMSprop/batch_normalization_10/beta/rms
4:2
û
ß2"RMSprop/input_dense2053/kernel/rms
-:+ß2 RMSprop/input_dense2053/bias/rms
1:/
ßý2RMSprop/mid_dense991/kernel/rms
*:(ý2RMSprop/mid_dense991/bias/rms
+:)	82RMSprop/dense_6/kernel/rms
$:"2RMSprop/dense_6/bias/rms
0:.	ým2RMSprop/mid_dense381/kernel/rms
):'m2RMSprop/mid_dense381/bias/rms
4:22(RMSprop/batch_normalization_11/gamma/rms
3:12'RMSprop/batch_normalization_11/beta/rms
/:-m2RMSprop/mid_dense109/kernel/rms
):'2RMSprop/mid_dense109/bias/rms
/:-2RMSprop/output_layer/kernel/rms
):'2RMSprop/output_layer/bias/rms
*:(2RMSprop/dense_7/kernel/rms
$:"2RMSprop/dense_7/bias/rms
*:(2RMSprop/dense_8/kernel/rms
$:"2RMSprop/dense_8/bias/rms
*:(2RMSprop/dense_9/kernel/rms
$:"2RMSprop/dense_9/bias/rms
#__inference__wrapped_model_15234561è:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓw¢t
m¢j
he
1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû

ª "1ª.
,
dense_9!
dense_9ÿÿÿÿÿÿÿÿÿ¦
J__inference_activation_4_layer_call_and_return_conditional_losses_15241789X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
/__inference_activation_4_layer_call_fn_15241784K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241098^_`aM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ï
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15241116^_`aM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ç
9__inference_batch_normalization_10_layer_call_fn_15241062^_`aM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ç
9__inference_batch_normalization_10_layer_call_fn_15241080^_`aM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¾
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241507f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15241541f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_11_layer_call_fn_15241453Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_11_layer_call_fn_15241487Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240890,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_15240908,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
8__inference_batch_normalization_8_layer_call_fn_15240854,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
8__inference_batch_normalization_8_layer_call_fn_15240872,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15240994EFGHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 î
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_15241012EFGHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Æ
8__inference_batch_normalization_9_layer_call_fn_15240958EFGHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Æ
8__inference_batch_normalization_9_layer_call_fn_15240976EFGHM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ó
K__inference_concatenate_1_layer_call_and_return_conditional_losses_15241803Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
0__inference_concatenate_1_layer_call_fn_15241796vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ·
F__inference_conv2d_6_layer_call_and_return_conditional_losses_15240836m#$8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿU
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿS~
 
+__inference_conv2d_6_layer_call_fn_15240825`#$8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿU
ª " ÿÿÿÿÿÿÿÿÿS~¶
F__inference_conv2d_7_layer_call_and_return_conditional_losses_15240940l<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ)?
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ'= 
 
+__inference_conv2d_7_layer_call_fn_15240929_<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ)?
ª " ÿÿÿÿÿÿÿÿÿ'= ¶
F__inference_conv2d_8_layer_call_and_return_conditional_losses_15241044lUV7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2d_8_layer_call_fn_15241033_UV7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¨
E__inference_dense_6_layer_call_and_return_conditional_losses_15241342_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ8
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_6_layer_call_fn_15241331R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ8
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_7_layer_call_and_return_conditional_losses_15241779^¶·/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_7_layer_call_fn_15241768Q¶·/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_8_layer_call_and_return_conditional_losses_15241825^ÊË/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_8_layer_call_fn_15241814QÊË/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_9_layer_call_and_return_conditional_losses_15241847^ÒÓ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_9_layer_call_fn_15241836QÒÓ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241654\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
G__inference_dropout_2_layer_call_and_return_conditional_losses_15241666\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dropout_2_layer_call_fn_15241637O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dropout_2_layer_call_fn_15241649O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¬
G__inference_flatten_2_layer_call_and_return_conditional_losses_15241229a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ8
 
,__inference_flatten_2_layer_call_fn_15241223T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ8c
9__inference_input_dense2053_activity_regularizer_15234832&¢
¢
	
x
ª " Á
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_15241217lno0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿû

ª "4¢1

0ÿÿÿÿÿÿÿÿÿß

	
1/0 ¯
M__inference_input_dense2053_layer_call_and_return_conditional_losses_15242034^no0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿû

ª "&¢#

0ÿÿÿÿÿÿÿÿÿß
 
2__inference_input_dense2053_layer_call_fn_15241179Qno0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿû

ª "ÿÿÿÿÿÿÿÿÿß=
__inference_loss_fn_0_15241867n¢

¢ 
ª " =
__inference_loss_fn_1_15241878o¢

¢ 
ª " =
__inference_loss_fn_2_15241898|¢

¢ 
ª " =
__inference_loss_fn_3_15241909}¢

¢ 
ª " >
__inference_loss_fn_4_15241929¢

¢ 
ª " >
__inference_loss_fn_5_15241940¢

¢ 
ª " >
__inference_loss_fn_6_15241960¢

¢ 
ª " >
__inference_loss_fn_7_15241971 ¢

¢ 
ª " >
__inference_loss_fn_8_15241991®¢

¢ 
ª " >
__inference_loss_fn_9_15242002¯¢

¢ 
ª " ð
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_15240918R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_6_layer_call_fn_15240913R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_15241022R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_7_layer_call_fn_15241017R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_15241126R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_8_layer_call_fn_15241121R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
6__inference_mid_dense109_activity_regularizer_15234981&¢
¢
	
x
ª " ¾
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_15241632l /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿm
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ¬
J__inference_mid_dense109_layer_call_and_return_conditional_losses_15242130^ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿm
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_mid_dense109_layer_call_fn_15241594Q /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿm
ª "ÿÿÿÿÿÿÿÿÿ`
6__inference_mid_dense381_activity_regularizer_15234858&¢
¢
	
x
ª " ¿
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_15241433m0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿý
ª "3¢0

0ÿÿÿÿÿÿÿÿÿm

	
1/0 ­
J__inference_mid_dense381_layer_call_and_return_conditional_losses_15242098_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿý
ª "%¢"

0ÿÿÿÿÿÿÿÿÿm
 
/__inference_mid_dense381_layer_call_fn_15241395R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿý
ª "ÿÿÿÿÿÿÿÿÿm`
6__inference_mid_dense991_activity_regularizer_15234845&¢
¢
	
x
ª " ¾
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_15241320l|}0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿß
ª "4¢1

0ÿÿÿÿÿÿÿÿÿý

	
1/0 ¬
J__inference_mid_dense991_layer_call_and_return_conditional_losses_15242066^|}0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿß
ª "&¢#

0ÿÿÿÿÿÿÿÿÿý
 
/__inference_mid_dense991_layer_call_fn_15241282Q|}0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿß
ª "ÿÿÿÿÿÿÿÿÿýô
E__inference_model_1_layer_call_and_return_conditional_losses_15238827ª:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓ¢|
u¢r
he
1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû

p 

 
ª "k¢h

0ÿÿÿÿÿÿÿÿÿ
IF
	
1/0 
	
1/1 
	
1/2 
	
1/3 
	
1/4 ô
E__inference_model_1_layer_call_and_return_conditional_losses_15239279ª:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓ¢|
u¢r
he
1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû

p

 
ª "k¢h

0ÿÿÿÿÿÿÿÿÿ
IF
	
1/0 
	
1/1 
	
1/2 
	
1/3 
	
1/4 á
E__inference_model_1_layer_call_and_return_conditional_losses_15240379:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓl¢i
b¢_
UR
+(
inputs/0ÿÿÿÿÿÿÿÿÿU
# 
inputs/1ÿÿÿÿÿÿÿÿÿû

p 

 
ª "k¢h

0ÿÿÿÿÿÿÿÿÿ
IF
	
1/0 
	
1/1 
	
1/2 
	
1/3 
	
1/4 á
E__inference_model_1_layer_call_and_return_conditional_losses_15240726:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓl¢i
b¢_
UR
+(
inputs/0ÿÿÿÿÿÿÿÿÿU
# 
inputs/1ÿÿÿÿÿÿÿÿÿû

p

 
ª "k¢h

0ÿÿÿÿÿÿÿÿÿ
IF
	
1/0 
	
1/1 
	
1/2 
	
1/3 
	
1/4 
*__inference_model_1_layer_call_fn_15235421×:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓ¢|
u¢r
he
1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_1_layer_call_fn_15238396×:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓ¢|
u¢r
he
1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû

p

 
ª "ÿÿÿÿÿÿÿÿÿó
*__inference_model_1_layer_call_fn_15239711Ä:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓl¢i
b¢_
UR
+(
inputs/0ÿÿÿÿÿÿÿÿÿU
# 
inputs/1ÿÿÿÿÿÿÿÿÿû

p 

 
ª "ÿÿÿÿÿÿÿÿÿó
*__inference_model_1_layer_call_fn_15240053Ä:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓl¢i
b¢_
UR
+(
inputs/0ÿÿÿÿÿÿÿÿÿU
# 
inputs/1ÿÿÿÿÿÿÿÿÿû

p

 
ª "ÿÿÿÿÿÿÿÿÿ`
6__inference_output_layer_activity_regularizer_15234994&¢
¢
	
x
ª " ¾
N__inference_output_layer_layer_call_and_return_all_conditional_losses_15241757l®¯/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ¬
J__inference_output_layer_layer_call_and_return_conditional_losses_15242162^®¯/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_output_layer_layer_call_fn_15241719Q®¯/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ½
&__inference_signature_wrapper_15240814:#$,-./<=EFGHUV^_`ano|} ®¯¶·ÊËÒÓ ¢
¢ 
ª
C
conv2d_6_input1.
conv2d_6_inputÿÿÿÿÿÿÿÿÿU
I
input_dense2053_input0-
input_dense2053_inputÿÿÿÿÿÿÿÿÿû
"1ª.
,
dense_9!
dense_9ÿÿÿÿÿÿÿÿÿ