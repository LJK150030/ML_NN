??!
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68?? 
?
input_dense2053/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*'
shared_nameinput_dense2053/kernel
?
*input_dense2053/kernel/Read/ReadVariableOpReadVariableOpinput_dense2053/kernel* 
_output_shapes
:
?
?*
dtype0
?
input_dense2053/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameinput_dense2053/bias
z
(input_dense2053/bias/Read/ReadVariableOpReadVariableOpinput_dense2053/bias*
_output_shapes	
:?*
dtype0
?
mid_dense991/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namemid_dense991/kernel
}
'mid_dense991/kernel/Read/ReadVariableOpReadVariableOpmid_dense991/kernel* 
_output_shapes
:
??*
dtype0
{
mid_dense991/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namemid_dense991/bias
t
%mid_dense991/bias/Read/ReadVariableOpReadVariableOpmid_dense991/bias*
_output_shapes	
:?*
dtype0
?
mid_dense381/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?m*$
shared_namemid_dense381/kernel
|
'mid_dense381/kernel/Read/ReadVariableOpReadVariableOpmid_dense381/kernel*
_output_shapes
:	?m*
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
?
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
?
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
"RMSprop/input_dense2053/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?
?*3
shared_name$"RMSprop/input_dense2053/kernel/rms
?
6RMSprop/input_dense2053/kernel/rms/Read/ReadVariableOpReadVariableOp"RMSprop/input_dense2053/kernel/rms* 
_output_shapes
:
?
?*
dtype0
?
 RMSprop/input_dense2053/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" RMSprop/input_dense2053/bias/rms
?
4RMSprop/input_dense2053/bias/rms/Read/ReadVariableOpReadVariableOp RMSprop/input_dense2053/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/mid_dense991/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*0
shared_name!RMSprop/mid_dense991/kernel/rms
?
3RMSprop/mid_dense991/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense991/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/mid_dense991/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameRMSprop/mid_dense991/bias/rms
?
1RMSprop/mid_dense991/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense991/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/mid_dense381/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?m*0
shared_name!RMSprop/mid_dense381/kernel/rms
?
3RMSprop/mid_dense381/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense381/kernel/rms*
_output_shapes
:	?m*
dtype0
?
RMSprop/mid_dense381/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:m*.
shared_nameRMSprop/mid_dense381/bias/rms
?
1RMSprop/mid_dense381/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense381/bias/rms*
_output_shapes
:m*
dtype0
?
RMSprop/mid_dense109/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m*0
shared_name!RMSprop/mid_dense109/kernel/rms
?
3RMSprop/mid_dense109/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense109/kernel/rms*
_output_shapes

:m*
dtype0
?
RMSprop/mid_dense109/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/mid_dense109/bias/rms
?
1RMSprop/mid_dense109/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mid_dense109/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/output_layer/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!RMSprop/output_layer/kernel/rms
?
3RMSprop/output_layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/kernel/rms*
_output_shapes

:*
dtype0
?
RMSprop/output_layer/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/output_layer/bias/rms
?
1RMSprop/output_layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output_layer/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
?B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?A
value?AB?A B?A
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
?
>iter
	?decay
@learning_rate
Amomentum
Brho
rms?
rms?
rms?
rms?
 rms?
!rms?
(rms?
)rms?
0rms?
1rms?*
J
0
1
2
3
 4
!5
(6
)7
08
19*
J
0
1
2
3
 4
!5
(6
)7
08
19*
H
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9* 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Rserving_default* 
f`
VARIABLE_VALUEinput_dense2053/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEinput_dense2053/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

C0
D1* 
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
Xactivity_regularizer_fn
*&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEmid_dense991/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmid_dense991/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

E0
F1* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_activity_regularizer_fn
*&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEmid_dense381/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmid_dense381/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*

G0
H1* 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
factivity_regularizer_fn
*'&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEmid_dense109/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmid_dense109/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*

I0
J1* 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
mactivity_regularizer_fn
*/&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*

K0
L1* 
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
tactivity_regularizer_fn
*7&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
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
* 
.
0
1
2
3
4
5*

{0
|1
}2*
* 
* 
* 
* 
* 
* 

C0
D1* 
* 
* 
* 
* 
* 
* 

E0
F1* 
* 
* 
* 
* 
* 
* 

G0
H1* 
* 
* 
* 
* 
* 
* 

I0
J1* 
* 
* 
* 
* 
* 
* 

K0
L1* 
* 
* 
* 
* 
* 
* 
* 
* 
:
	~total
	count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUE"RMSprop/input_dense2053/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE RMSprop/input_dense2053/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense991/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense991/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense381/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense381/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense109/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/mid_dense109/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/output_layer/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/output_layer/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?
%serving_default_input_dense2053_inputPlaceholder*(
_output_shapes
:??????????
*
dtype0*
shape:??????????

?
StatefulPartitionedCallStatefulPartitionedCall%serving_default_input_dense2053_inputinput_dense2053/kernelinput_dense2053/biasmid_dense991/kernelmid_dense991/biasmid_dense381/kernelmid_dense381/biasmid_dense109/kernelmid_dense109/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_35563912
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
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices*input_dense2053/kernel/Read/ReadVariableOp(input_dense2053/bias/Read/ReadVariableOp'mid_dense991/kernel/Read/ReadVariableOp%mid_dense991/bias/Read/ReadVariableOp'mid_dense381/kernel/Read/ReadVariableOp%mid_dense381/bias/Read/ReadVariableOp'mid_dense109/kernel/Read/ReadVariableOp%mid_dense109/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp6RMSprop/input_dense2053/kernel/rms/Read/ReadVariableOp4RMSprop/input_dense2053/bias/rms/Read/ReadVariableOp3RMSprop/mid_dense991/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense991/bias/rms/Read/ReadVariableOp3RMSprop/mid_dense381/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense381/bias/rms/Read/ReadVariableOp3RMSprop/mid_dense109/kernel/rms/Read/ReadVariableOp1RMSprop/mid_dense109/bias/rms/Read/ReadVariableOp3RMSprop/output_layer/kernel/rms/Read/ReadVariableOp1RMSprop/output_layer/bias/rms/Read/ReadVariableOpConst"/device:CPU:0*.
dtypes$
"2 	
?
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
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOpAssignVariableOpinput_dense2053/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_1AssignVariableOpinput_dense2053/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_2AssignVariableOpmid_dense991/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_3AssignVariableOpmid_dense991/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_4AssignVariableOpmid_dense381/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_5AssignVariableOpmid_dense381/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_6AssignVariableOpmid_dense109/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_7AssignVariableOpmid_dense109/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_8AssignVariableOpoutput_layer/kernel
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_9AssignVariableOpoutput_layer/biasIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0	*
_output_shapes
:
^
AssignVariableOp_10AssignVariableOpRMSprop/iterIdentity_11"/device:CPU:0*
dtype0	
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_11AssignVariableOpRMSprop/decayIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_12AssignVariableOpRMSprop/learning_rateIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_13AssignVariableOpRMSprop/momentumIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_14AssignVariableOpRMSprop/rhoIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_15AssignVariableOptotalIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_16AssignVariableOpcountIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_17AssignVariableOptotal_1Identity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_18AssignVariableOpcount_1Identity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_19AssignVariableOptotal_2Identity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_20AssignVariableOpcount_2Identity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_21AssignVariableOp"RMSprop/input_dense2053/kernel/rmsIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_22AssignVariableOp RMSprop/input_dense2053/bias/rmsIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_23AssignVariableOpRMSprop/mid_dense991/kernel/rmsIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_24AssignVariableOpRMSprop/mid_dense991/bias/rmsIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_25AssignVariableOpRMSprop/mid_dense381/kernel/rmsIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_26AssignVariableOpRMSprop/mid_dense381/bias/rmsIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_27AssignVariableOpRMSprop/mid_dense109/kernel/rmsIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_28AssignVariableOpRMSprop/mid_dense109/bias/rmsIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_29AssignVariableOpRMSprop/output_layer/kernel/rmsIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_30AssignVariableOpRMSprop/output_layer/bias/rmsIdentity_31"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_32Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ǿ
?
?
__inference_loss_fn_3_35564439K
<mid_dense991_bias_regularizer_square_readvariableop_resource:	?
identity??3mid_dense991/bias/Regularizer/Square/ReadVariableOp?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense991_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
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
?+
?
N__inference_output_layer_layer_call_and_return_all_conditional_losses_35564367

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: O
SquareSquareSigmoid:y:0*
T0*'
_output_shapes
:?????????V
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
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
̗
?

#__inference__wrapped_model_35559840
input_dense2053_inputP
<sequential_36_input_dense2053_matmul_readvariableop_resource:
?
?L
=sequential_36_input_dense2053_biasadd_readvariableop_resource:	?M
9sequential_36_mid_dense991_matmul_readvariableop_resource:
??I
:sequential_36_mid_dense991_biasadd_readvariableop_resource:	?L
9sequential_36_mid_dense381_matmul_readvariableop_resource:	?mH
:sequential_36_mid_dense381_biasadd_readvariableop_resource:mK
9sequential_36_mid_dense109_matmul_readvariableop_resource:mH
:sequential_36_mid_dense109_biasadd_readvariableop_resource:K
9sequential_36_output_layer_matmul_readvariableop_resource:H
:sequential_36_output_layer_biasadd_readvariableop_resource:
identity??4sequential_36/input_dense2053/BiasAdd/ReadVariableOp?3sequential_36/input_dense2053/MatMul/ReadVariableOp?1sequential_36/mid_dense109/BiasAdd/ReadVariableOp?0sequential_36/mid_dense109/MatMul/ReadVariableOp?1sequential_36/mid_dense381/BiasAdd/ReadVariableOp?0sequential_36/mid_dense381/MatMul/ReadVariableOp?1sequential_36/mid_dense991/BiasAdd/ReadVariableOp?0sequential_36/mid_dense991/MatMul/ReadVariableOp?1sequential_36/output_layer/BiasAdd/ReadVariableOp?0sequential_36/output_layer/MatMul/ReadVariableOp?
3sequential_36/input_dense2053/MatMul/ReadVariableOpReadVariableOp<sequential_36_input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
$sequential_36/input_dense2053/MatMulMatMulinput_dense2053_input;sequential_36/input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
4sequential_36/input_dense2053/BiasAdd/ReadVariableOpReadVariableOp=sequential_36_input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential_36/input_dense2053/BiasAddBiasAdd.sequential_36/input_dense2053/MatMul:product:0<sequential_36/input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
"sequential_36/input_dense2053/ReluRelu.sequential_36/input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8sequential_36/input_dense2053/ActivityRegularizer/SquareSquare0sequential_36/input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:???????????
7sequential_36/input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5sequential_36/input_dense2053/ActivityRegularizer/SumSum<sequential_36/input_dense2053/ActivityRegularizer/Square:y:0@sequential_36/input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: |
7sequential_36/input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
5sequential_36/input_dense2053/ActivityRegularizer/mulMul@sequential_36/input_dense2053/ActivityRegularizer/mul/x:output:0>sequential_36/input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
7sequential_36/input_dense2053/ActivityRegularizer/ShapeShape0sequential_36/input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
Esequential_36/input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_36/input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_36/input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_36/input_dense2053/ActivityRegularizer/strided_sliceStridedSlice@sequential_36/input_dense2053/ActivityRegularizer/Shape:output:0Nsequential_36/input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Psequential_36/input_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Psequential_36/input_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_36/input_dense2053/ActivityRegularizer/CastCastHsequential_36/input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
9sequential_36/input_dense2053/ActivityRegularizer/truedivRealDiv9sequential_36/input_dense2053/ActivityRegularizer/mul:z:0:sequential_36/input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
0sequential_36/mid_dense991/MatMul/ReadVariableOpReadVariableOp9sequential_36_mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
!sequential_36/mid_dense991/MatMulMatMul0sequential_36/input_dense2053/Relu:activations:08sequential_36/mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
1sequential_36/mid_dense991/BiasAdd/ReadVariableOpReadVariableOp:sequential_36_mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"sequential_36/mid_dense991/BiasAddBiasAdd+sequential_36/mid_dense991/MatMul:product:09sequential_36/mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_36/mid_dense991/ReluRelu+sequential_36/mid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
5sequential_36/mid_dense991/ActivityRegularizer/SquareSquare-sequential_36/mid_dense991/Relu:activations:0*
T0*(
_output_shapes
:???????????
4sequential_36/mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
2sequential_36/mid_dense991/ActivityRegularizer/SumSum9sequential_36/mid_dense991/ActivityRegularizer/Square:y:0=sequential_36/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: y
4sequential_36/mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
2sequential_36/mid_dense991/ActivityRegularizer/mulMul=sequential_36/mid_dense991/ActivityRegularizer/mul/x:output:0;sequential_36/mid_dense991/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
4sequential_36/mid_dense991/ActivityRegularizer/ShapeShape-sequential_36/mid_dense991/Relu:activations:0*
T0*
_output_shapes
:?
Bsequential_36/mid_dense991/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_36/mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_36/mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_36/mid_dense991/ActivityRegularizer/strided_sliceStridedSlice=sequential_36/mid_dense991/ActivityRegularizer/Shape:output:0Ksequential_36/mid_dense991/ActivityRegularizer/strided_slice/stack:output:0Msequential_36/mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0Msequential_36/mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3sequential_36/mid_dense991/ActivityRegularizer/CastCastEsequential_36/mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
6sequential_36/mid_dense991/ActivityRegularizer/truedivRealDiv6sequential_36/mid_dense991/ActivityRegularizer/mul:z:07sequential_36/mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
0sequential_36/mid_dense381/MatMul/ReadVariableOpReadVariableOp9sequential_36_mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
!sequential_36/mid_dense381/MatMulMatMul-sequential_36/mid_dense991/Relu:activations:08sequential_36/mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
1sequential_36/mid_dense381/BiasAdd/ReadVariableOpReadVariableOp:sequential_36_mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
"sequential_36/mid_dense381/BiasAddBiasAdd+sequential_36/mid_dense381/MatMul:product:09sequential_36/mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
sequential_36/mid_dense381/ReluRelu+sequential_36/mid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m?
5sequential_36/mid_dense381/ActivityRegularizer/SquareSquare-sequential_36/mid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????m?
4sequential_36/mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
2sequential_36/mid_dense381/ActivityRegularizer/SumSum9sequential_36/mid_dense381/ActivityRegularizer/Square:y:0=sequential_36/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: y
4sequential_36/mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
2sequential_36/mid_dense381/ActivityRegularizer/mulMul=sequential_36/mid_dense381/ActivityRegularizer/mul/x:output:0;sequential_36/mid_dense381/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
4sequential_36/mid_dense381/ActivityRegularizer/ShapeShape-sequential_36/mid_dense381/Relu:activations:0*
T0*
_output_shapes
:?
Bsequential_36/mid_dense381/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_36/mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_36/mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_36/mid_dense381/ActivityRegularizer/strided_sliceStridedSlice=sequential_36/mid_dense381/ActivityRegularizer/Shape:output:0Ksequential_36/mid_dense381/ActivityRegularizer/strided_slice/stack:output:0Msequential_36/mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0Msequential_36/mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3sequential_36/mid_dense381/ActivityRegularizer/CastCastEsequential_36/mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
6sequential_36/mid_dense381/ActivityRegularizer/truedivRealDiv6sequential_36/mid_dense381/ActivityRegularizer/mul:z:07sequential_36/mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
0sequential_36/mid_dense109/MatMul/ReadVariableOpReadVariableOp9sequential_36_mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
!sequential_36/mid_dense109/MatMulMatMul-sequential_36/mid_dense381/Relu:activations:08sequential_36/mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_36/mid_dense109/BiasAdd/ReadVariableOpReadVariableOp:sequential_36_mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"sequential_36/mid_dense109/BiasAddBiasAdd+sequential_36/mid_dense109/MatMul:product:09sequential_36/mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_36/mid_dense109/ReluRelu+sequential_36/mid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
5sequential_36/mid_dense109/ActivityRegularizer/SquareSquare-sequential_36/mid_dense109/Relu:activations:0*
T0*'
_output_shapes
:??????????
4sequential_36/mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
2sequential_36/mid_dense109/ActivityRegularizer/SumSum9sequential_36/mid_dense109/ActivityRegularizer/Square:y:0=sequential_36/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: y
4sequential_36/mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
2sequential_36/mid_dense109/ActivityRegularizer/mulMul=sequential_36/mid_dense109/ActivityRegularizer/mul/x:output:0;sequential_36/mid_dense109/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
4sequential_36/mid_dense109/ActivityRegularizer/ShapeShape-sequential_36/mid_dense109/Relu:activations:0*
T0*
_output_shapes
:?
Bsequential_36/mid_dense109/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_36/mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_36/mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_36/mid_dense109/ActivityRegularizer/strided_sliceStridedSlice=sequential_36/mid_dense109/ActivityRegularizer/Shape:output:0Ksequential_36/mid_dense109/ActivityRegularizer/strided_slice/stack:output:0Msequential_36/mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0Msequential_36/mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3sequential_36/mid_dense109/ActivityRegularizer/CastCastEsequential_36/mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
6sequential_36/mid_dense109/ActivityRegularizer/truedivRealDiv6sequential_36/mid_dense109/ActivityRegularizer/mul:z:07sequential_36/mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
0sequential_36/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_36_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
!sequential_36/output_layer/MatMulMatMul-sequential_36/mid_dense109/Relu:activations:08sequential_36/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_36/output_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_36_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"sequential_36/output_layer/BiasAddBiasAdd+sequential_36/output_layer/MatMul:product:09sequential_36/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"sequential_36/output_layer/SigmoidSigmoid+sequential_36/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
5sequential_36/output_layer/ActivityRegularizer/SquareSquare&sequential_36/output_layer/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
4sequential_36/output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
2sequential_36/output_layer/ActivityRegularizer/SumSum9sequential_36/output_layer/ActivityRegularizer/Square:y:0=sequential_36/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: y
4sequential_36/output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
2sequential_36/output_layer/ActivityRegularizer/mulMul=sequential_36/output_layer/ActivityRegularizer/mul/x:output:0;sequential_36/output_layer/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
4sequential_36/output_layer/ActivityRegularizer/ShapeShape&sequential_36/output_layer/Sigmoid:y:0*
T0*
_output_shapes
:?
Bsequential_36/output_layer/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_36/output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_36/output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_36/output_layer/ActivityRegularizer/strided_sliceStridedSlice=sequential_36/output_layer/ActivityRegularizer/Shape:output:0Ksequential_36/output_layer/ActivityRegularizer/strided_slice/stack:output:0Msequential_36/output_layer/ActivityRegularizer/strided_slice/stack_1:output:0Msequential_36/output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3sequential_36/output_layer/ActivityRegularizer/CastCastEsequential_36/output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
6sequential_36/output_layer/ActivityRegularizer/truedivRealDiv6sequential_36/output_layer/ActivityRegularizer/mul:z:07sequential_36/output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
#sequential_36/activation_11/SoftmaxSoftmax&sequential_36/output_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????|
IdentityIdentity-sequential_36/activation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp5^sequential_36/input_dense2053/BiasAdd/ReadVariableOp4^sequential_36/input_dense2053/MatMul/ReadVariableOp2^sequential_36/mid_dense109/BiasAdd/ReadVariableOp1^sequential_36/mid_dense109/MatMul/ReadVariableOp2^sequential_36/mid_dense381/BiasAdd/ReadVariableOp1^sequential_36/mid_dense381/MatMul/ReadVariableOp2^sequential_36/mid_dense991/BiasAdd/ReadVariableOp1^sequential_36/mid_dense991/MatMul/ReadVariableOp2^sequential_36/output_layer/BiasAdd/ReadVariableOp1^sequential_36/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2l
4sequential_36/input_dense2053/BiasAdd/ReadVariableOp4sequential_36/input_dense2053/BiasAdd/ReadVariableOp2j
3sequential_36/input_dense2053/MatMul/ReadVariableOp3sequential_36/input_dense2053/MatMul/ReadVariableOp2f
1sequential_36/mid_dense109/BiasAdd/ReadVariableOp1sequential_36/mid_dense109/BiasAdd/ReadVariableOp2d
0sequential_36/mid_dense109/MatMul/ReadVariableOp0sequential_36/mid_dense109/MatMul/ReadVariableOp2f
1sequential_36/mid_dense381/BiasAdd/ReadVariableOp1sequential_36/mid_dense381/BiasAdd/ReadVariableOp2d
0sequential_36/mid_dense381/MatMul/ReadVariableOp0sequential_36/mid_dense381/MatMul/ReadVariableOp2f
1sequential_36/mid_dense991/BiasAdd/ReadVariableOp1sequential_36/mid_dense991/BiasAdd/ReadVariableOp2d
0sequential_36/mid_dense991/MatMul/ReadVariableOp0sequential_36/mid_dense991/MatMul/ReadVariableOp2f
1sequential_36/output_layer/BiasAdd/ReadVariableOp1sequential_36/output_layer/BiasAdd/ReadVariableOp2d
0sequential_36/output_layer/MatMul/ReadVariableOp0sequential_36/output_layer/MatMul/ReadVariableOp:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?+
?
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_35564094

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: W
SquareSquareRelu:activations:0*
T0*(
_output_shapes
:??????????V
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
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
P
9__inference_input_dense2053_activity_regularizer_35559853
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
:?????????G
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
 *??'7I
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
?'
?
J__inference_mid_dense991_layer_call_and_return_conditional_losses_35564596

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
6__inference_mid_dense109_activity_regularizer_35559892
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
:?????????G
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
 *??'7I
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
?'
?
/__inference_mid_dense991_layer_call_fn_35564056

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?'
?
J__inference_mid_dense109_layer_call_and_return_conditional_losses_35564660

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_35563912
input_dense2053_input
unknown:
?
?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?m
	unknown_4:m
	unknown_5:m
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_dense2053_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_35559840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?+
?
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_35564185

inputs1
matmul_readvariableop_resource:	?m-
biasadd_readvariableop_resource:m
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????mj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: V
SquareSquareRelu:activations:0*
T0*'
_output_shapes
:?????????mV
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
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????mG

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_35564428O
;mid_dense991_kernel_regularizer_abs_readvariableop_resource:
??
identity??2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense991_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense991_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense991/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
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
ܿ
?
0__inference_sequential_36_layer_call_fn_35560216
input_dense2053_inputB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp?Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp??mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp??mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp??mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp??output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:??????????}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?z
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????mw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?'
?
J__inference_mid_dense381_layer_call_and_return_conditional_losses_35564628

inputs1
matmul_readvariableop_resource:	?m-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????mj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_activation_11_layer_call_and_return_conditional_losses_35564377

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_35564470J
<mid_dense381_bias_regularizer_square_readvariableop_resource:m
identity??3mid_dense381/bias/Regularizer/Square/ReadVariableOp?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense381_bias_regularizer_square_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
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
??
?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562629
input_dense2053_inputB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp?Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp??mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp??mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp??mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp??output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:??????????}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?z
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????mw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

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
: ?
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?
?
__inference_loss_fn_9_35564532J
<output_layer_bias_regularizer_square_readvariableop_resource:
identity??3output_layer/bias/Regularizer/Square/ReadVariableOp?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp<output_layer_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
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
?'
?
J__inference_output_layer_layer_call_and_return_conditional_losses_35564692

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
/__inference_mid_dense109_layer_call_fn_35564238

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
?(
?
2__inference_input_dense2053_layer_call_fn_35563965

inputs2
matmul_readvariableop_resource:
?
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?&
?
/__inference_output_layer_layer_call_fn_35564329

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563885

inputsB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinputs-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

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
: ?	
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
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
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_35564397R
>input_dense2053_kernel_regularizer_abs_readvariableop_resource:
?
?
identity??5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOpm
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>input_dense2053_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>input_dense2053_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentity,input_dense2053/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
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
?
M
6__inference_mid_dense991_activity_regularizer_35559866
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
:?????????G
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
 *??'7I
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
?
?
__inference_loss_fn_4_35564459N
;mid_dense381_kernel_regularizer_abs_readvariableop_resource:	?m
identity??2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense381_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense381_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense381/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp
??
?
0__inference_sequential_36_layer_call_fn_35563260

inputsB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinputs-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
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
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_7_35564501J
<mid_dense109_bias_regularizer_square_readvariableop_resource:
identity??3mid_dense109/bias/Regularizer/Square/ReadVariableOp?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp<mid_dense109_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
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
?
L
0__inference_activation_11_layer_call_fn_35564372

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
0__inference_sequential_36_layer_call_fn_35563465

inputsB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinputs-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
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
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
M
6__inference_mid_dense381_activity_regularizer_35559879
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
:?????????G
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
 *??'7I
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
??
?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562944
input_dense2053_inputB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp?Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp??mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp??mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp??mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp??output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:??????????}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?z
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????mw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

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
: ?
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?+
?
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_35564276

inputs0
matmul_readvariableop_resource:m-
biasadd_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: V
SquareSquareRelu:activations:0*
T0*'
_output_shapes
:?????????V
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
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????m
 
_user_specified_nameinputs
??
?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563675

inputsB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinputs-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

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
: ?	
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
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
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?(
?
M__inference_input_dense2053_layer_call_and_return_conditional_losses_35564564

inputs2
matmul_readvariableop_resource:
?
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_8_35564521M
;output_layer_kernel_regularizer_abs_readvariableop_resource:
identity??2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOpj
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;output_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;output_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)output_layer/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
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
?
?
__inference_loss_fn_6_35564490M
;mid_dense109_kernel_regularizer_abs_readvariableop_resource:m
identity??2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOpj
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;mid_dense109_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;mid_dense109_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
IdentityIdentity)mid_dense109/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
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
ܿ
?
0__inference_sequential_36_layer_call_fn_35562314
input_dense2053_inputB
.input_dense2053_matmul_readvariableop_resource:
?
?>
/input_dense2053_biasadd_readvariableop_resource:	??
+mid_dense991_matmul_readvariableop_resource:
??;
,mid_dense991_biasadd_readvariableop_resource:	?>
+mid_dense381_matmul_readvariableop_resource:	?m:
,mid_dense381_biasadd_readvariableop_resource:m=
+mid_dense109_matmul_readvariableop_resource:m:
,mid_dense109_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity??&input_dense2053/BiasAdd/ReadVariableOp?%input_dense2053/MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp?Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOp?#mid_dense109/BiasAdd/ReadVariableOp?"mid_dense109/MatMul/ReadVariableOp?3mid_dense109/bias/Regularizer/Square/ReadVariableOp?2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense109/kernel/Regularizer/Square/ReadVariableOp?@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp??mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp?#mid_dense381/BiasAdd/ReadVariableOp?"mid_dense381/MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOp?@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp??mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp?#mid_dense991/BiasAdd/ReadVariableOp?"mid_dense991/MatMul/ReadVariableOp?3mid_dense991/bias/Regularizer/Square/ReadVariableOp?2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense991/kernel/Regularizer/Square/ReadVariableOp?@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp??mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?3output_layer/bias/Regularizer/Square/ReadVariableOp?2output_layer/kernel/Regularizer/Abs/ReadVariableOp?5output_layer/kernel/Regularizer/Square/ReadVariableOp?@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp??output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp?
%input_dense2053/MatMul/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
input_dense2053/MatMulMatMulinput_dense2053_input-input_dense2053/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&input_dense2053/BiasAdd/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
input_dense2053/BiasAddBiasAdd input_dense2053/MatMul:product:0.input_dense2053/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
input_dense2053/ReluRelu input_dense2053/BiasAdd:output:0*
T0*(
_output_shapes
:??????????}
8input_dense2053/input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
6input_dense2053/input_dense2053/kernel/Regularizer/AbsAbsMinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6input_dense2053/input_dense2053/kernel/Regularizer/SumSum:input_dense2053/input_dense2053/kernel/Regularizer/Abs:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: }
8input_dense2053/input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
6input_dense2053/input_dense2053/kernel/Regularizer/mulMulAinput_dense2053/input_dense2053/kernel/Regularizer/mul/x:output:0?input_dense2053/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
6input_dense2053/input_dense2053/kernel/Regularizer/addAddV2Ainput_dense2053/input_dense2053/kernel/Regularizer/Const:output:0:input_dense2053/input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
9input_dense2053/input_dense2053/kernel/Regularizer/SquareSquarePinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
??
:input_dense2053/input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
8input_dense2053/input_dense2053/kernel/Regularizer/Sum_1Sum=input_dense2053/input_dense2053/kernel/Regularizer/Square:y:0Cinput_dense2053/input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 
:input_dense2053/input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
8input_dense2053/input_dense2053/kernel/Regularizer/mul_1MulCinput_dense2053/input_dense2053/kernel/Regularizer/mul_1/x:output:0Ainput_dense2053/input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
8input_dense2053/input_dense2053/kernel/Regularizer/add_1AddV2:input_dense2053/input_dense2053/kernel/Regularizer/add:z:0<input_dense2053/input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7input_dense2053/input_dense2053/bias/Regularizer/SquareSquareNinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
6input_dense2053/input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4input_dense2053/input_dense2053/bias/Regularizer/SumSum;input_dense2053/input_dense2053/bias/Regularizer/Square:y:0?input_dense2053/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: {
6input_dense2053/input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
4input_dense2053/input_dense2053/bias/Regularizer/mulMul?input_dense2053/input_dense2053/bias/Regularizer/mul/x:output:0=input_dense2053/input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*input_dense2053/ActivityRegularizer/SquareSquare"input_dense2053/Relu:activations:0*
T0*(
_output_shapes
:??????????z
)input_dense2053/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
'input_dense2053/ActivityRegularizer/SumSum.input_dense2053/ActivityRegularizer/Square:y:02input_dense2053/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)input_dense2053/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
'input_dense2053/ActivityRegularizer/mulMul2input_dense2053/ActivityRegularizer/mul/x:output:00input_dense2053/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: {
)input_dense2053/ActivityRegularizer/ShapeShape"input_dense2053/Relu:activations:0*
T0*
_output_shapes
:?
7input_dense2053/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9input_dense2053/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9input_dense2053/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1input_dense2053/ActivityRegularizer/strided_sliceStridedSlice2input_dense2053/ActivityRegularizer/Shape:output:0@input_dense2053/ActivityRegularizer/strided_slice/stack:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_1:output:0Binput_dense2053/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(input_dense2053/ActivityRegularizer/CastCast:input_dense2053/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
+input_dense2053/ActivityRegularizer/truedivRealDiv+input_dense2053/ActivityRegularizer/mul:z:0,input_dense2053/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense991/MatMul/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
mid_dense991/MatMulMatMul"input_dense2053/Relu:activations:0*mid_dense991/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#mid_dense991/BiasAdd/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
mid_dense991/BiasAddBiasAddmid_dense991/MatMul:product:0+mid_dense991/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
mid_dense991/ReluRelumid_dense991/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
2mid_dense991/mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
0mid_dense991/mid_dense991/kernel/Regularizer/AbsAbsGmid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense991/mid_dense991/kernel/Regularizer/SumSum4mid_dense991/mid_dense991/kernel/Regularizer/Abs:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense991/mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense991/mid_dense991/kernel/Regularizer/mulMul;mid_dense991/mid_dense991/kernel/Regularizer/mul/x:output:09mid_dense991/mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense991/mid_dense991/kernel/Regularizer/addAddV2;mid_dense991/mid_dense991/kernel/Regularizer/Const:output:04mid_dense991/mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3mid_dense991/mid_dense991/kernel/Regularizer/SquareSquareJmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
4mid_dense991/mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense991/mid_dense991/kernel/Regularizer/Sum_1Sum7mid_dense991/mid_dense991/kernel/Regularizer/Square:y:0=mid_dense991/mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense991/mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense991/mid_dense991/kernel/Regularizer/mul_1Mul=mid_dense991/mid_dense991/kernel/Regularizer/mul_1/x:output:0;mid_dense991/mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense991/mid_dense991/kernel/Regularizer/add_1AddV24mid_dense991/mid_dense991/kernel/Regularizer/add:z:06mid_dense991/mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1mid_dense991/mid_dense991/bias/Regularizer/SquareSquareHmid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?z
0mid_dense991/mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense991/mid_dense991/bias/Regularizer/SumSum5mid_dense991/mid_dense991/bias/Regularizer/Square:y:09mid_dense991/mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense991/mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense991/mid_dense991/bias/Regularizer/mulMul9mid_dense991/mid_dense991/bias/Regularizer/mul/x:output:07mid_dense991/mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense991/ActivityRegularizer/SquareSquaremid_dense991/Relu:activations:0*
T0*(
_output_shapes
:??????????w
&mid_dense991/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense991/ActivityRegularizer/SumSum+mid_dense991/ActivityRegularizer/Square:y:0/mid_dense991/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense991/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense991/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense991/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense991/ActivityRegularizer/strided_sliceStridedSlice/mid_dense991/ActivityRegularizer/Shape:output:0=mid_dense991/ActivityRegularizer/strided_slice/stack:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense991/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense991/ActivityRegularizer/CastCast7mid_dense991/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense991/ActivityRegularizer/truedivRealDiv(mid_dense991/ActivityRegularizer/mul:z:0)mid_dense991/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense381/MatMul/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
mid_dense381/MatMulMatMulmid_dense991/Relu:activations:0*mid_dense381/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m?
#mid_dense381/BiasAdd/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
mid_dense381/BiasAddBiasAddmid_dense381/MatMul:product:0+mid_dense381/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mj
mid_dense381/ReluRelumid_dense381/BiasAdd:output:0*
T0*'
_output_shapes
:?????????mw
2mid_dense381/mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
0mid_dense381/mid_dense381/kernel/Regularizer/AbsAbsGmid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense381/mid_dense381/kernel/Regularizer/SumSum4mid_dense381/mid_dense381/kernel/Regularizer/Abs:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense381/mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense381/mid_dense381/kernel/Regularizer/mulMul;mid_dense381/mid_dense381/kernel/Regularizer/mul/x:output:09mid_dense381/mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense381/mid_dense381/kernel/Regularizer/addAddV2;mid_dense381/mid_dense381/kernel/Regularizer/Const:output:04mid_dense381/mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
3mid_dense381/mid_dense381/kernel/Regularizer/SquareSquareJmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?m?
4mid_dense381/mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense381/mid_dense381/kernel/Regularizer/Sum_1Sum7mid_dense381/mid_dense381/kernel/Regularizer/Square:y:0=mid_dense381/mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense381/mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense381/mid_dense381/kernel/Regularizer/mul_1Mul=mid_dense381/mid_dense381/kernel/Regularizer/mul_1/x:output:0;mid_dense381/mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense381/mid_dense381/kernel/Regularizer/add_1AddV24mid_dense381/mid_dense381/kernel/Regularizer/add:z:06mid_dense381/mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
1mid_dense381/mid_dense381/bias/Regularizer/SquareSquareHmid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mz
0mid_dense381/mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense381/mid_dense381/bias/Regularizer/SumSum5mid_dense381/mid_dense381/bias/Regularizer/Square:y:09mid_dense381/mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense381/mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense381/mid_dense381/bias/Regularizer/mulMul9mid_dense381/mid_dense381/bias/Regularizer/mul/x:output:07mid_dense381/mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense381/ActivityRegularizer/SquareSquaremid_dense381/Relu:activations:0*
T0*'
_output_shapes
:?????????mw
&mid_dense381/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense381/ActivityRegularizer/SumSum+mid_dense381/ActivityRegularizer/Square:y:0/mid_dense381/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense381/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense381/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense381/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense381/ActivityRegularizer/strided_sliceStridedSlice/mid_dense381/ActivityRegularizer/Shape:output:0=mid_dense381/ActivityRegularizer/strided_slice/stack:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense381/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense381/ActivityRegularizer/CastCast7mid_dense381/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense381/ActivityRegularizer/truedivRealDiv(mid_dense381/ActivityRegularizer/mul:z:0)mid_dense381/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"mid_dense109/MatMul/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
mid_dense109/MatMulMatMulmid_dense381/Relu:activations:0*mid_dense109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#mid_dense109/BiasAdd/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
mid_dense109/BiasAddBiasAddmid_dense109/MatMul:product:0+mid_dense109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
mid_dense109/ReluRelumid_dense109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2mid_dense109/mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
0mid_dense109/mid_dense109/kernel/Regularizer/AbsAbsGmid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0mid_dense109/mid_dense109/kernel/Regularizer/SumSum4mid_dense109/mid_dense109/kernel/Regularizer/Abs:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2mid_dense109/mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0mid_dense109/mid_dense109/kernel/Regularizer/mulMul;mid_dense109/mid_dense109/kernel/Regularizer/mul/x:output:09mid_dense109/mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0mid_dense109/mid_dense109/kernel/Regularizer/addAddV2;mid_dense109/mid_dense109/kernel/Regularizer/Const:output:04mid_dense109/mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
3mid_dense109/mid_dense109/kernel/Regularizer/SquareSquareJmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:m?
4mid_dense109/mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2mid_dense109/mid_dense109/kernel/Regularizer/Sum_1Sum7mid_dense109/mid_dense109/kernel/Regularizer/Square:y:0=mid_dense109/mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4mid_dense109/mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2mid_dense109/mid_dense109/kernel/Regularizer/mul_1Mul=mid_dense109/mid_dense109/kernel/Regularizer/mul_1/x:output:0;mid_dense109/mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2mid_dense109/mid_dense109/kernel/Regularizer/add_1AddV24mid_dense109/mid_dense109/kernel/Regularizer/add:z:06mid_dense109/mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1mid_dense109/mid_dense109/bias/Regularizer/SquareSquareHmid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0mid_dense109/mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.mid_dense109/mid_dense109/bias/Regularizer/SumSum5mid_dense109/mid_dense109/bias/Regularizer/Square:y:09mid_dense109/mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0mid_dense109/mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.mid_dense109/mid_dense109/bias/Regularizer/mulMul9mid_dense109/mid_dense109/bias/Regularizer/mul/x:output:07mid_dense109/mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'mid_dense109/ActivityRegularizer/SquareSquaremid_dense109/Relu:activations:0*
T0*'
_output_shapes
:?????????w
&mid_dense109/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$mid_dense109/ActivityRegularizer/SumSum+mid_dense109/ActivityRegularizer/Square:y:0/mid_dense109/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&mid_dense109/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6mid_dense109/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mid_dense109/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mid_dense109/ActivityRegularizer/strided_sliceStridedSlice/mid_dense109/ActivityRegularizer/Shape:output:0=mid_dense109/ActivityRegularizer/strided_slice/stack:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_1:output:0?mid_dense109/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%mid_dense109/ActivityRegularizer/CastCast7mid_dense109/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(mid_dense109/ActivityRegularizer/truedivRealDiv(mid_dense109/ActivityRegularizer/mul:z:0)mid_dense109/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output_layer/MatMulMatMulmid_dense109/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
2output_layer/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0output_layer/output_layer/kernel/Regularizer/AbsAbsGoutput_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0output_layer/output_layer/kernel/Regularizer/SumSum4output_layer/output_layer/kernel/Regularizer/Abs:y:0=output_layer/output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: w
2output_layer/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0output_layer/output_layer/kernel/Regularizer/mulMul;output_layer/output_layer/kernel/Regularizer/mul/x:output:09output_layer/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0output_layer/output_layer/kernel/Regularizer/addAddV2;output_layer/output_layer/kernel/Regularizer/Const:output:04output_layer/output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3output_layer/output_layer/kernel/Regularizer/SquareSquareJoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
4output_layer/output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
2output_layer/output_layer/kernel/Regularizer/Sum_1Sum7output_layer/output_layer/kernel/Regularizer/Square:y:0=output_layer/output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: y
4output_layer/output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
2output_layer/output_layer/kernel/Regularizer/mul_1Mul=output_layer/output_layer/kernel/Regularizer/mul_1/x:output:0;output_layer/output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
2output_layer/output_layer/kernel/Regularizer/add_1AddV24output_layer/output_layer/kernel/Regularizer/add:z:06output_layer/output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1output_layer/output_layer/bias/Regularizer/SquareSquareHoutput_layer/output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:z
0output_layer/output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.output_layer/output_layer/bias/Regularizer/SumSum5output_layer/output_layer/bias/Regularizer/Square:y:09output_layer/output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: u
0output_layer/output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
.output_layer/output_layer/bias/Regularizer/mulMul9output_layer/output_layer/bias/Regularizer/mul/x:output:07output_layer/output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
'output_layer/ActivityRegularizer/SquareSquareoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????w
&output_layer/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
$output_layer/ActivityRegularizer/SumSum+output_layer/ActivityRegularizer/Square:y:0/output_layer/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&output_layer/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
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
valueB: ?
6output_layer/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6output_layer/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.output_layer/ActivityRegularizer/strided_sliceStridedSlice/output_layer/ActivityRegularizer/Shape:output:0=output_layer/ActivityRegularizer/strided_slice/stack:output:0?output_layer/ActivityRegularizer/strided_slice/stack_1:output:0?output_layer/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%output_layer/ActivityRegularizer/CastCast7output_layer/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(output_layer/ActivityRegularizer/truedivRealDiv(output_layer/ActivityRegularizer/mul:z:0)output_layer/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: l
activation_11/SoftmaxSoftmaxoutput_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.input_dense2053_matmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp/input_dense2053_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#mid_dense991/kernel/Regularizer/AbsAbs:mid_dense991/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense991/kernel/Regularizer/SumSum'mid_dense991/kernel/Regularizer/Abs:y:00mid_dense991/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense991/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense991/kernel/Regularizer/mulMul.mid_dense991/kernel/Regularizer/mul/x:output:0,mid_dense991/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense991/kernel/Regularizer/addAddV2.mid_dense991/kernel/Regularizer/Const:output:0'mid_dense991/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense991/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense991_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
&mid_dense991/kernel/Regularizer/SquareSquare=mid_dense991/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??x
'mid_dense991/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense991/kernel/Regularizer/Sum_1Sum*mid_dense991/kernel/Regularizer/Square:y:00mid_dense991/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense991/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense991/kernel/Regularizer/mul_1Mul0mid_dense991/kernel/Regularizer/mul_1/x:output:0.mid_dense991/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense991/kernel/Regularizer/add_1AddV2'mid_dense991/kernel/Regularizer/add:z:0)mid_dense991/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense991/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense991_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$mid_dense991/bias/Regularizer/SquareSquare;mid_dense991/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?m
#mid_dense991/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense991/bias/Regularizer/SumSum(mid_dense991/bias/Regularizer/Square:y:0,mid_dense991/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense991/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense991/bias/Regularizer/mulMul,mid_dense991/bias/Regularizer/mul/x:output:0*mid_dense991/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense381_matmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense381_biasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
#mid_dense109/kernel/Regularizer/AbsAbs:mid_dense109/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense109/kernel/Regularizer/SumSum'mid_dense109/kernel/Regularizer/Abs:y:00mid_dense109/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense109/kernel/Regularizer/mulMul.mid_dense109/kernel/Regularizer/mul/x:output:0,mid_dense109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense109/kernel/Regularizer/addAddV2.mid_dense109/kernel/Regularizer/Const:output:0'mid_dense109/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+mid_dense109_matmul_readvariableop_resource*
_output_shapes

:m*
dtype0?
&mid_dense109/kernel/Regularizer/SquareSquare=mid_dense109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:mx
'mid_dense109/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense109/kernel/Regularizer/Sum_1Sum*mid_dense109/kernel/Regularizer/Square:y:00mid_dense109/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense109/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense109/kernel/Regularizer/mul_1Mul0mid_dense109/kernel/Regularizer/mul_1/x:output:0.mid_dense109/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense109/kernel/Regularizer/add_1AddV2'mid_dense109/kernel/Regularizer/add:z:0)mid_dense109/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense109/bias/Regularizer/Square/ReadVariableOpReadVariableOp,mid_dense109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$mid_dense109/bias/Regularizer/SquareSquare;mid_dense109/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#mid_dense109/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense109/bias/Regularizer/SumSum(mid_dense109/bias/Regularizer/Square:y:0,mid_dense109/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense109/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense109/bias/Regularizer/mulMul,mid_dense109/bias/Regularizer/mul/x:output:0*mid_dense109/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2output_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#output_layer/kernel/Regularizer/AbsAbs:output_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#output_layer/kernel/Regularizer/SumSum'output_layer/kernel/Regularizer/Abs:y:00output_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0,output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#output_layer/kernel/Regularizer/addAddV2.output_layer/kernel/Regularizer/Const:output:0'output_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
&output_layer/kernel/Regularizer/SquareSquare=output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:x
'output_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%output_layer/kernel/Regularizer/Sum_1Sum*output_layer/kernel/Regularizer/Square:y:00output_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'output_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%output_layer/kernel/Regularizer/mul_1Mul0output_layer/kernel/Regularizer/mul_1/x:output:0.output_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%output_layer/kernel/Regularizer/add_1AddV2'output_layer/kernel/Regularizer/add:z:0)output_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3output_layer/bias/Regularizer/Square/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$output_layer/bias/Regularizer/SquareSquare;output_layer/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#output_layer/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!output_layer/bias/Regularizer/SumSum(output_layer/bias/Regularizer/Square:y:0,output_layer/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#output_layer/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!output_layer/bias/Regularizer/mulMul,output_layer/bias/Regularizer/mul/x:output:0*output_layer/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^input_dense2053/BiasAdd/ReadVariableOp&^input_dense2053/MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOpG^input_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpF^input_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpI^input_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp$^mid_dense109/BiasAdd/ReadVariableOp#^mid_dense109/MatMul/ReadVariableOp4^mid_dense109/bias/Regularizer/Square/ReadVariableOp3^mid_dense109/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense109/kernel/Regularizer/Square/ReadVariableOpA^mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@^mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp$^mid_dense381/BiasAdd/ReadVariableOp#^mid_dense381/MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOpA^mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@^mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp$^mid_dense991/BiasAdd/ReadVariableOp#^mid_dense991/MatMul/ReadVariableOp4^mid_dense991/bias/Regularizer/Square/ReadVariableOp3^mid_dense991/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense991/kernel/Regularizer/Square/ReadVariableOpA^mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@^mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOpC^mid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp4^output_layer/bias/Regularizer/Square/ReadVariableOp3^output_layer/kernel/Regularizer/Abs/ReadVariableOp6^output_layer/kernel/Regularizer/Square/ReadVariableOpA^output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@^output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOpC^output_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????
: : : : : : : : : : 2P
&input_dense2053/BiasAdd/ReadVariableOp&input_dense2053/BiasAdd/ReadVariableOp2N
%input_dense2053/MatMul/ReadVariableOp%input_dense2053/MatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Finput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOpFinput_dense2053/input_dense2053/bias/Regularizer/Square/ReadVariableOp2?
Einput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOpEinput_dense2053/input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2?
Hinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOpHinput_dense2053/input_dense2053/kernel/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense109/BiasAdd/ReadVariableOp#mid_dense109/BiasAdd/ReadVariableOp2H
"mid_dense109/MatMul/ReadVariableOp"mid_dense109/MatMul/ReadVariableOp2j
3mid_dense109/bias/Regularizer/Square/ReadVariableOp3mid_dense109/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense109/kernel/Regularizer/Square/ReadVariableOp5mid_dense109/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp@mid_dense109/mid_dense109/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp?mid_dense109/mid_dense109/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOpBmid_dense109/mid_dense109/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense381/BiasAdd/ReadVariableOp#mid_dense381/BiasAdd/ReadVariableOp2H
"mid_dense381/MatMul/ReadVariableOp"mid_dense381/MatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp@mid_dense381/mid_dense381/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?mid_dense381/mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOpBmid_dense381/mid_dense381/kernel/Regularizer/Square/ReadVariableOp2J
#mid_dense991/BiasAdd/ReadVariableOp#mid_dense991/BiasAdd/ReadVariableOp2H
"mid_dense991/MatMul/ReadVariableOp"mid_dense991/MatMul/ReadVariableOp2j
3mid_dense991/bias/Regularizer/Square/ReadVariableOp3mid_dense991/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense991/kernel/Regularizer/Square/ReadVariableOp5mid_dense991/kernel/Regularizer/Square/ReadVariableOp2?
@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp@mid_dense991/mid_dense991/bias/Regularizer/Square/ReadVariableOp2?
?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp?mid_dense991/mid_dense991/kernel/Regularizer/Abs/ReadVariableOp2?
Bmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOpBmid_dense991/mid_dense991/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2j
3output_layer/bias/Regularizer/Square/ReadVariableOp3output_layer/bias/Regularizer/Square/ReadVariableOp2h
2output_layer/kernel/Regularizer/Abs/ReadVariableOp2output_layer/kernel/Regularizer/Abs/ReadVariableOp2n
5output_layer/kernel/Regularizer/Square/ReadVariableOp5output_layer/kernel/Regularizer/Square/ReadVariableOp2?
@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp@output_layer/output_layer/bias/Regularizer/Square/ReadVariableOp2?
?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp?output_layer/output_layer/kernel/Regularizer/Abs/ReadVariableOp2?
Boutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOpBoutput_layer/output_layer/kernel/Regularizer/Square/ReadVariableOp:_ [
(
_output_shapes
:??????????

/
_user_specified_nameinput_dense2053_input
?&
?
/__inference_mid_dense381_layer_call_fn_35564147

inputs1
matmul_readvariableop_resource:	?m-
biasadd_readvariableop_resource:m
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3mid_dense381/bias/Regularizer/Square/ReadVariableOp?2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp?5mid_dense381/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????mP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????mj
%mid_dense381/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
#mid_dense381/kernel/Regularizer/AbsAbs:mid_dense381/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
#mid_dense381/kernel/Regularizer/SumSum'mid_dense381/kernel/Regularizer/Abs:y:00mid_dense381/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: j
%mid_dense381/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#mid_dense381/kernel/Regularizer/mulMul.mid_dense381/kernel/Regularizer/mul/x:output:0,mid_dense381/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
#mid_dense381/kernel/Regularizer/addAddV2.mid_dense381/kernel/Regularizer/Const:output:0'mid_dense381/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
5mid_dense381/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?m*
dtype0?
&mid_dense381/kernel/Regularizer/SquareSquare=mid_dense381/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?mx
'mid_dense381/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
%mid_dense381/kernel/Regularizer/Sum_1Sum*mid_dense381/kernel/Regularizer/Square:y:00mid_dense381/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: l
'mid_dense381/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
%mid_dense381/kernel/Regularizer/mul_1Mul0mid_dense381/kernel/Regularizer/mul_1/x:output:0.mid_dense381/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
%mid_dense381/kernel/Regularizer/add_1AddV2'mid_dense381/kernel/Regularizer/add:z:0)mid_dense381/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
3mid_dense381/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:m*
dtype0?
$mid_dense381/bias/Regularizer/SquareSquare;mid_dense381/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:mm
#mid_dense381/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!mid_dense381/bias/Regularizer/SumSum(mid_dense381/bias/Regularizer/Square:y:0,mid_dense381/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#mid_dense381/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!mid_dense381/bias/Regularizer/mulMul,mid_dense381/bias/Regularizer/mul/x:output:0*mid_dense381/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????m?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^mid_dense381/bias/Regularizer/Square/ReadVariableOp3^mid_dense381/kernel/Regularizer/Abs/ReadVariableOp6^mid_dense381/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3mid_dense381/bias/Regularizer/Square/ReadVariableOp3mid_dense381/bias/Regularizer/Square/ReadVariableOp2h
2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2mid_dense381/kernel/Regularizer/Abs/ReadVariableOp2n
5mid_dense381/kernel/Regularizer/Square/ReadVariableOp5mid_dense381/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_35564408N
?input_dense2053_bias_regularizer_square_readvariableop_resource:	?
identity??6input_dense2053/bias/Regularizer/Square/ReadVariableOp?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOp?input_dense2053_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
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
?,
?
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_35564003

inputs2
matmul_readvariableop_resource:
?
?.
biasadd_readvariableop_resource:	?
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?6input_dense2053/bias/Regularizer/Square/ReadVariableOp?5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp?8input_dense2053/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????m
(input_dense2053/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
&input_dense2053/kernel/Regularizer/AbsAbs=input_dense2053/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&input_dense2053/kernel/Regularizer/SumSum*input_dense2053/kernel/Regularizer/Abs:y:03input_dense2053/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(input_dense2053/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&input_dense2053/kernel/Regularizer/mulMul1input_dense2053/kernel/Regularizer/mul/x:output:0/input_dense2053/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&input_dense2053/kernel/Regularizer/addAddV21input_dense2053/kernel/Regularizer/Const:output:0*input_dense2053/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8input_dense2053/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?
?*
dtype0?
)input_dense2053/kernel/Regularizer/SquareSquare@input_dense2053/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?
?{
*input_dense2053/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(input_dense2053/kernel/Regularizer/Sum_1Sum-input_dense2053/kernel/Regularizer/Square:y:03input_dense2053/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*input_dense2053/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(input_dense2053/kernel/Regularizer/mul_1Mul3input_dense2053/kernel/Regularizer/mul_1/x:output:01input_dense2053/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(input_dense2053/kernel/Regularizer/add_1AddV2*input_dense2053/kernel/Regularizer/add:z:0,input_dense2053/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
6input_dense2053/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'input_dense2053/bias/Regularizer/SquareSquare>input_dense2053/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?p
&input_dense2053/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$input_dense2053/bias/Regularizer/SumSum+input_dense2053/bias/Regularizer/Square:y:0/input_dense2053/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&input_dense2053/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$input_dense2053/bias/Regularizer/mulMul/input_dense2053/bias/Regularizer/mul/x:output:0-input_dense2053/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: W
SquareSquareRelu:activations:0*
T0*(
_output_shapes
:??????????V
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
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^input_dense2053/bias/Regularizer/Square/ReadVariableOp6^input_dense2053/kernel/Regularizer/Abs/ReadVariableOp9^input_dense2053/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6input_dense2053/bias/Regularizer/Square/ReadVariableOp6input_dense2053/bias/Regularizer/Square/ReadVariableOp2n
5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp5input_dense2053/kernel/Regularizer/Abs/ReadVariableOp2t
8input_dense2053/kernel/Regularizer/Square/ReadVariableOp8input_dense2053/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
M
6__inference_output_layer_activity_regularizer_35559905
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
:?????????G
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
 *??'7I
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
_user_specified_namex"?-
saver_filename:0
Identity:0Identity_328"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
input_dense2053_input?
'serving_default_input_dense2053_input:0??????????
A
activation_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>iter
	?decay
@learning_rate
Amomentum
Brho
rms?
rms?
rms?
rms?
 rms?
!rms?
(rms?
)rms?
0rms?
1rms?"
	optimizer
f
0
1
2
3
 4
!5
(6
)7
08
19"
trackable_list_wrapper
f
0
1
2
3
 4
!5
(6
)7
08
19"
trackable_list_wrapper
f
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9"
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_sequential_36_layer_call_fn_35560216
0__inference_sequential_36_layer_call_fn_35563260
0__inference_sequential_36_layer_call_fn_35563465
0__inference_sequential_36_layer_call_fn_35562314?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563675
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563885
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562629
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562944?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_35559840input_dense2053_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Rserving_default"
signature_map
*:(
?
?2input_dense2053/kernel
#:!?2input_dense2053/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
Xactivity_regularizer_fn
*&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_input_dense2053_layer_call_fn_35563965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_35564003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%
??2mid_dense991/kernel
 :?2mid_dense991/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_activity_regularizer_fn
*&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_mid_dense991_layer_call_fn_35564056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_35564094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
&:$	?m2mid_dense381/kernel
:m2mid_dense381/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
factivity_regularizer_fn
*'&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_mid_dense381_layer_call_fn_35564147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_35564185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
%:#m2mid_dense109/kernel
:2mid_dense109/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
mactivity_regularizer_fn
*/&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_mid_dense109_layer_call_fn_35564238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_35564276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
%:#2output_layer/kernel
:2output_layer/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
tactivity_regularizer_fn
*7&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_output_layer_layer_call_fn_35564329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_output_layer_layer_call_and_return_all_conditional_losses_35564367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_activation_11_layer_call_fn_35564372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_activation_11_layer_call_and_return_conditional_losses_35564377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
?2?
__inference_loss_fn_0_35564397?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_35564408?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_35564428?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_35564439?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_35564459?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_35564470?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_35564490?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_35564501?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_8_35564521?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_9_35564532?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
{0
|1
}2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_signature_wrapper_35563912input_dense2053_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
9__inference_input_dense2053_activity_regularizer_35559853?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
M__inference_input_dense2053_layer_call_and_return_conditional_losses_35564564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
6__inference_mid_dense991_activity_regularizer_35559866?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
J__inference_mid_dense991_layer_call_and_return_conditional_losses_35564596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
6__inference_mid_dense381_activity_regularizer_35559879?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
J__inference_mid_dense381_layer_call_and_return_conditional_losses_35564628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
6__inference_mid_dense109_activity_regularizer_35559892?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
J__inference_mid_dense109_layer_call_and_return_conditional_losses_35564660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
6__inference_output_layer_activity_regularizer_35559905?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
J__inference_output_layer_layer_call_and_return_conditional_losses_35564692?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
P
	~total
	count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
4:2
?
?2"RMSprop/input_dense2053/kernel/rms
-:+?2 RMSprop/input_dense2053/bias/rms
1:/
??2RMSprop/mid_dense991/kernel/rms
*:(?2RMSprop/mid_dense991/bias/rms
0:.	?m2RMSprop/mid_dense381/kernel/rms
):'m2RMSprop/mid_dense381/bias/rms
/:-m2RMSprop/mid_dense109/kernel/rms
):'2RMSprop/mid_dense109/bias/rms
/:-2RMSprop/output_layer/kernel/rms
):'2RMSprop/output_layer/bias/rms?
#__inference__wrapped_model_35559840?
 !()01??<
5?2
0?-
input_dense2053_input??????????

? "=?:
8
activation_11'?$
activation_11??????????
K__inference_activation_11_layer_call_and_return_conditional_losses_35564377X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
0__inference_activation_11_layer_call_fn_35564372K/?,
%?"
 ?
inputs?????????
? "??????????c
9__inference_input_dense2053_activity_regularizer_35559853&?
?
?	
x
? "? ?
Q__inference_input_dense2053_layer_call_and_return_all_conditional_losses_35564003l0?-
&?#
!?
inputs??????????

? "4?1
?
0??????????
?
?	
1/0 ?
M__inference_input_dense2053_layer_call_and_return_conditional_losses_35564564^0?-
&?#
!?
inputs??????????

? "&?#
?
0??????????
? ?
2__inference_input_dense2053_layer_call_fn_35563965Q0?-
&?#
!?
inputs??????????

? "???????????=
__inference_loss_fn_0_35564397?

? 
? "? =
__inference_loss_fn_1_35564408?

? 
? "? =
__inference_loss_fn_2_35564428?

? 
? "? =
__inference_loss_fn_3_35564439?

? 
? "? =
__inference_loss_fn_4_35564459 ?

? 
? "? =
__inference_loss_fn_5_35564470!?

? 
? "? =
__inference_loss_fn_6_35564490(?

? 
? "? =
__inference_loss_fn_7_35564501)?

? 
? "? =
__inference_loss_fn_8_355645210?

? 
? "? =
__inference_loss_fn_9_355645321?

? 
? "? `
6__inference_mid_dense109_activity_regularizer_35559892&?
?
?	
x
? "? ?
N__inference_mid_dense109_layer_call_and_return_all_conditional_losses_35564276j()/?,
%?"
 ?
inputs?????????m
? "3?0
?
0?????????
?
?	
1/0 ?
J__inference_mid_dense109_layer_call_and_return_conditional_losses_35564660\()/?,
%?"
 ?
inputs?????????m
? "%?"
?
0?????????
? ?
/__inference_mid_dense109_layer_call_fn_35564238O()/?,
%?"
 ?
inputs?????????m
? "??????????`
6__inference_mid_dense381_activity_regularizer_35559879&?
?
?	
x
? "? ?
N__inference_mid_dense381_layer_call_and_return_all_conditional_losses_35564185k !0?-
&?#
!?
inputs??????????
? "3?0
?
0?????????m
?
?	
1/0 ?
J__inference_mid_dense381_layer_call_and_return_conditional_losses_35564628] !0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????m
? ?
/__inference_mid_dense381_layer_call_fn_35564147P !0?-
&?#
!?
inputs??????????
? "??????????m`
6__inference_mid_dense991_activity_regularizer_35559866&?
?
?	
x
? "? ?
N__inference_mid_dense991_layer_call_and_return_all_conditional_losses_35564094l0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_mid_dense991_layer_call_and_return_conditional_losses_35564596^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_mid_dense991_layer_call_fn_35564056Q0?-
&?#
!?
inputs??????????
? "???????????`
6__inference_output_layer_activity_regularizer_35559905&?
?
?	
x
? "? ?
N__inference_output_layer_layer_call_and_return_all_conditional_losses_35564367j01/?,
%?"
 ?
inputs?????????
? "3?0
?
0?????????
?
?	
1/0 ?
J__inference_output_layer_layer_call_and_return_conditional_losses_35564692\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
/__inference_output_layer_layer_call_fn_35564329O01/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562629?
 !()01G?D
=?:
0?-
input_dense2053_input??????????

p 

 
? "k?h
?
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35562944?
 !()01G?D
=?:
0?-
input_dense2053_input??????????

p

 
? "k?h
?
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563675?
 !()018?5
.?+
!?
inputs??????????

p 

 
? "k?h
?
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
K__inference_sequential_36_layer_call_and_return_conditional_losses_35563885?
 !()018?5
.?+
!?
inputs??????????

p

 
? "k?h
?
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
0__inference_sequential_36_layer_call_fn_35560216o
 !()01G?D
=?:
0?-
input_dense2053_input??????????

p 

 
? "???????????
0__inference_sequential_36_layer_call_fn_35562314o
 !()01G?D
=?:
0?-
input_dense2053_input??????????

p

 
? "???????????
0__inference_sequential_36_layer_call_fn_35563260`
 !()018?5
.?+
!?
inputs??????????

p 

 
? "???????????
0__inference_sequential_36_layer_call_fn_35563465`
 !()018?5
.?+
!?
inputs??????????

p

 
? "???????????
&__inference_signature_wrapper_35563912?
 !()01X?U
? 
N?K
I
input_dense2053_input0?-
input_dense2053_input??????????
"=?:
8
activation_11'?$
activation_11?????????