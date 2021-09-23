from glob import glob
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from generic_UNet import InitWeights_He
import pickle
import torch.nn.functional as F
from generic_UNet import Generic_UNet

prefix = "version5"

planfile = "/data/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/2d/Task002_ABDSeg/nnUNetTrainer__nnUNetPlansv2.1/plans.pkl"
modelfile = "/data/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/2d/Task002_ABDSeg/nnUNetTrainer__nnUNetPlansv2.1/all/model_final_checkpoint.model"
rawfs = glob("/home/SENSETIME/luoxiangde.vendor/Projects/ABDSeg/data/ABDSeg/data/imagesTs/*.nii.gz")

info = pickle.load(open(planfile, "rb"))
plan_data = {}
plan_data["plans"] = info
print(plan_data)


def recycle_plot(data, prefix):
    for k, v in data.items():
        if isinstance(v, dict):
            if isinstance(k, int):
                k = "%d" % k
            recycle_plot(v, prefix + "->" + k)
        else:
            print(prefix, k, v)


print("Inference")
resolution_index = 1
num_classes = plan_data['plans']['num_classes']
base_num_features = plan_data['plans']['base_num_features']
patch_size = plan_data['plans']['plans_per_stage'][resolution_index]['patch_size']
pool_op_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index]['pool_op_kernel_sizes']
conv_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index]['conv_kernel_sizes']
current_spacing = plan_data['plans']['plans_per_stage'][resolution_index]['current_spacing']

mean = plan_data['plans']['dataset_properties']['intensityproperties'][0]['mean']
std = plan_data['plans']['dataset_properties']['intensityproperties'][0]['sd']
clip_min = plan_data['plans']['dataset_properties']['intensityproperties'][0]['percentile_00_5']
clip_max = plan_data['plans']['dataset_properties']['intensityproperties'][0]['percentile_99_5']

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net = Generic_UNet(1, base_num_features, num_classes + 1, len(pool_op_kernel_sizes), 2, 2,
                   nn.Conv2d, nn.InstanceNorm2d, norm_op_kwargs, nn.Dropout2d,
                   dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
                   InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes, False, True, True)
net.cuda()
checkpoint = torch.load(modelfile)
weights = checkpoint['state_dict']
net.load_state_dict(weights, strict=False)
net.eval()
net.half()


def _get_arr(path):
    sitkimg = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitkimg)
    return arr, sitkimg


def _write_arr(arr, path, info=None):
    sitkimg = sitk.GetImageFromArray(arr)
    if info is not None:
        sitkimg.CopyInformation(info)
    sitk.WriteImage(sitkimg, path)


def get_do_separate_z(spacing, anisotropy_threshold=2):
    # do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    do_separate_z = spacing[-1] > anisotropy_threshold
    return do_separate_z


def predict2D(arr, batch_size=4):
    prob_map = torch.zeros((1, num_classes + 1,) + arr.shape).half().cuda()
    arr_clip = np.clip(arr, clip_min, clip_max)
    raw_norm = (arr_clip - mean) / std
    ind_x = np.array([i for i in range(raw_norm.shape[0])])
    for ind in ind_x[::batch_size]:
        print(ind)
        if ind + batch_size < raw_norm.shape[0]:
            tensor_arr = torch.from_numpy(raw_norm[ind:ind + batch_size, ...]).cuda().half().unsqueeze(1)
            with torch.no_grad():
                seg_pro = net(tensor_arr)
                _pred = seg_pro
                prob_map[:, :, ind:ind + batch_size, ...] += _pred.permute(1, 0, 2, 3)
        else:
            tensor_arr = torch.from_numpy(raw_norm[ind:, ...]).cuda().half().unsqueeze(1)
            with torch.no_grad():
                seg_pro = net(tensor_arr)
                _pred = seg_pro
                prob_map[:, :, ind:, ...] += _pred.permute(1, 0, 2, 3)
    torch.cuda.empty_cache()
    return prob_map.detach().cpu()


def itk_change_spacing(src_itk, output_spacing, interpolate_method='Linear'):
    assert interpolate_method in ['Linear', 'NearestNeighbor']
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()

    re_sample_scale = tuple(np.array(src_spacing) / np.array(output_spacing).astype(np.float))
    re_sample_size = tuple(np.array(src_size).astype(np.float) * np.array(re_sample_scale))

    re_sample_size = [int(round(x)) for x in re_sample_size]
    output_spacing = tuple((np.array(src_size) / np.array(re_sample_size)) * np.array(src_spacing))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetOutputPixelType(src_itk.GetPixelID())
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(re_sample_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    return re_sampler.Execute(src_itk)


def resample_image_to_ref(image, ref, interp=sitk.sitkNearestNeighbor, pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(interp)
    return resample.Execute(image)


def Inference2D(rawf):
    arr_raw, sitk_raw = _get_arr(rawf)
    origin_spacing = sitk_raw.GetSpacing()
    img_arr = arr_raw

    prob_map = predict2D(img_arr)

    if get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1]):
        print('postpreprocessing: do seperate z......')
        prob_map_interp_xy = torch.zeros(
            list(prob_map.size()[:2]) + [prob_map.size()[2], ] + list(sitk_raw.GetSize()[::-1][1:]), dtype=torch.half)

        for i in range(prob_map.size(2)):
            prob_map_interp_xy[:, :, i] = F.interpolate(prob_map[:, :, i].cuda().float(),
                                                        size=sitk_raw.GetSize()[::-1][1:],
                                                        mode="bilinear").detach().half().cpu()
        del prob_map

        prob_map_interp = np.zeros(list(prob_map_interp_xy.size()[:2]) + list(sitk_raw.GetSize()[::-1]),
                                   dtype=np.float16)

        for i in range(prob_map_interp.shape[1]):
            prob_map_interp[:, i] = F.interpolate(prob_map_interp_xy[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="nearest").detach().half().cpu().numpy()
        del prob_map_interp_xy

    else:
        prob_map_interp = np.zeros(list(prob_map.size()[:2]) + list(sitk_raw.GetSize()[::-1]), dtype=np.float16)

        for i in range(prob_map.size(1)):
            prob_map_interp[:, i] = F.interpolate(prob_map[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="trilinear").detach().half().cpu().numpy()
        del prob_map

    vessel_clf = np.argmax(prob_map_interp.squeeze(0), axis=0)
    del prob_map_interp

    pred_sitk = sitk.GetImageFromArray(vessel_clf.astype(np.uint8))
    pred_sitk.CopyInformation(sitk_raw)
    pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)
    sitk.WriteImage(pred_sitk, rawf.replace(".nii.gz", "_nnUNet2D_pred.nii.gz"))


rawf = "/home/SENSETIME/luoxiangde.vendor/Projects/ABDSeg/data/ABDSeg/data/imagesTs/ABD_0014.nii.gz"
Inference2D(rawf)
