import os
import glob
import shutil
import pickle

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from operator import attrgetter
from tqdm import tqdm
from pathlib import Path
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.cam_anim import create_image_as_png, _ffmpeg_high_quality, _ffmpeg_standard_quality, count_parameters

import time
import json

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:


        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")
    
    def cam_anim(self,
                img,
                img_tensor,
                norm_type='global', # TODO: implement "both" 
                frame_rate=5, 
                tmp_dir='tmp_anim', 
                output_fname='output.mp4',
                keep_frames=False,
                overlay=True, # TODO: implement this
                quality='standard'):
        
        """ cam_anim
            TODO: General description here
            --- general parameters
            model <PyTorch.nn>, the pretrained model used to generate layer-wise activations
            architecture_type <str>, the type of model architecture;
            options: ['densenet', 'resnet', 'vgg', ...]
            cam_func_name <str>, the exact name of the CAM class to use;
            options: ['GradCAM', 'HiResCAM', ...]
            global_layer_norm=True, normalize accross all layers for smooth transition;
            if False, we perform layer-spefic normalization
            include_final_map=True <bool>, optionally includes the final layer map
            return_layer_name_map=False <bool>, optionally returns a list of the layer name map (network vs. ordered frame) 
            use_cuda=True <bool>, TODO

            --- storage-specific parameters
            tmp_dir='tmp_anim' <str, Path>, os.path.join(os.getcwd(), 'tmp_anim')
            output_fname='output.mp4' <str, Path>, path to where the output file is saved
            keep_frames=False <bool>, determines whether or not the interim activation frames are retained or deleted
            --- ffmpeg-specific parameters
            frame_rate=5,

            --- animation-specific parameters
            quality='standard' <str>, determines what level of quality to render the video via ffmpeg: ['standard', 'high']
            include_title=True <bool>, adds the layer name to each picture title
            colormap='jet'

        """
        
        start = time.time()
        metrics_log = {"n_parameters": count_parameters(self.model), "layers_records": []}

        # cast the tmp_dir to string-representation (if passed as a Path object) TODO: test this with PATHS
        if not tmp_dir=='tmp_anim': tmp_dir = str(tmp_dir)
        
        # ensure that the path ends with a terminal os.sep (OS-agnostic)
        if not tmp_dir.endswith(os.sep): tmp_dir += os.sep
        
        # create the tmp_dir where images will be stored (clean out if already exists)
        if not os.path.exists(tmp_dir):
            try:
                os.mkdir(tmp_dir)
            except: 
                print('ERROR: failed to create tmp_dir: ' + tmp_dir + '. Exitting.')                
                return None
        else: # clean out any existing images if the dir already exists
            # for f in glob.glob(tmp_dir + '*'): os.remove(f)
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

        # EMILY ------------------------------------  
        # Generate & save images/arrays for all layers; save them to tmp_dir

        count = 0
        layer_name_map = {}
        temp_dict = {}
        
        # store init parameters
        reset_norm = False
        if hasattr(self, "normalization") and self.normalization:
            reset_norm = True
        self.normalization = False
        init_target_layers = self.target_layers

        # TODO: activations and gradients are large. Need to dump this to storage instead to releave memory
        with open(tmp_dir+"init_activations_and_grads.pkl", 'wb') as f:
            pickle.dump(self.activations_and_grads, f)

        mx = None
        mn = None

        # n_layers = 0
        # for layer_name, layer_module in self.model.named_modules():
        #     n_layers += 1
        layer_names = [layer_name for layer_name, _ in self.model.named_modules()]
        pbar = tqdm(layer_names, desc="Model Layer Loop")
        for layer_name in pbar:
            pbar.set_description(f"Model Layer Loop - {layer_name}")
            pbar.refresh()

            layer_record = {"layer_name": layer_name, "layer_id": str("%06d"%count), "error": None}
            layer_start_time = time.time()
            try:
                layer_module = attrgetter(layer_name)(self.model)
                layer_record.update({"layer_num_parameters": count_parameters(layer_module)})
                self.target_layers = [layer_module]
                self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, self.reshape_transform)
                # print(layer, self.target_layers[0])
                cam = self.__call__(input_tensor=img_tensor, targets=None) # for now targets was always None...

                # get global max value
                # if mx is None:
                #     mx = np.max(cam)
                # else:
                #     layer_mx = np.max(cam)
                #     if mx < layer_mx:
                #         mx = layer_mx
                
                # # get global max value
                # if mn is None:
                #     mn = np.min(cam)
                # else:
                #     layer_mn = np.min(cam)
                #     if mn > layer_mn:
                #         mn = layer_mn
                
                # TODO: save unnormalized cam to file instead of storing to temp_dict?

                # store cam to temp_dict
                temp_dict[str("%06d"%count)] = cam
                layer_name_map[str("%06d"%count)] = layer_name
                count += 1
                self.activations_and_grads.release()
                del self.activations_and_grads

                # save layer time
                layer_end_time = time.time()
                layer_record["layer_time"] = layer_end_time-layer_start_time
            except Exception as ex:
                layer_record["error"] = str(ex)
                # TODO: add more informative thing here
                print('skipping ' + layer_name)
                print(ex)
            
            metrics_log["layers_records"].append(layer_record)

        # reset init state
        if reset_norm:
            self.normalization = True
        self.target_layers = init_target_layers

        with open(tmp_dir+"init_activations_and_grads.pkl", "rb") as f:
            self.activations_and_grads = pickle.load(f)
        # self.activations_and_grads = init_activations_and_grads
        os.remove(tmp_dir+"init_activations_and_grads.pkl")
    
        # normalize img
        if np.max(img) > 1:
            img = np.float32(img) / np.max(img)
        
        # normalize cam
        mx = np.max(np.concatenate(list(temp_dict.values())))
        mn = np.min(np.concatenate(list(temp_dict.values())))
        for layer_id, cam in temp_dict.items():
            # norm_type is 'both' , we normalize laer-wise first, then globally. This ensures that the layer normalization is not affected by the global normalization
            if norm_type == 'layer' or norm_type == 'both':
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
                create_image_as_png(img, layer_id, cam, layer_name_map, tmp_dir+'layer')
            if norm_type == 'global' or norm_type == 'both':
                cam = (cam - mn) / (mx - mn)
                create_image_as_png(img, layer_id, cam, layer_name_map, tmp_dir+'global')
            
        

        # O(L*nm), n=width, m=height
        # EMILY ------------------------------------
        
        # if the output_fname already exists, ffmpeg will fail, so we overwrite it by deleting original.
        if os.path.exists(output_fname): os.remove(output_fname)

        # generate the animation; automatically saves to file
        if norm_type == 'both':    
            frames_dir = {"global": tmp_dir+os.sep+'global', "layer": tmp_dir+os.sep+'layer'}
        else: frames_dir = {norm_type: tmp_dir+os.sep+norm_type}

        for n_type, fr_dir in frames_dir.items():
            # path semantics
            Path(fr_dir).mkdir(parents=True, exist_ok=True)
            output_fname = Path(output_fname)
            output_fname = output_fname.parent / (output_fname.stem + f"_{n_type}" + output_fname.suffix)
            output_fname = str(output_fname)

            if quality =='high': _ffmpeg_high_quality(fr_dir, output_fname, frame_rate=frame_rate)
            else: _ffmpeg_standard_quality(fr_dir, output_fname, frame_rate=frame_rate)

        # if we do not keep the individual frames, we remove the tmp_dir & contents
        if not keep_frames: shutil.rmtree(tmp_dir)

        # return the list of two-tuples mapping original layer name to new filename

        end = time.time()
        metrics_log["cam_anim_time"] =  end-start


        # with open("sample.json", "w") as outfile:
        #     outfile.write(metrics_log)


        return metrics_log


    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        """_summary_

        Args:
            input_tensor (torch.Tensor): _description_
            targets (List[torch.nn.Module]): _description_
            eigen_smooth (bool): _description_

        Returns:
            np.ndarray: _description_
        """

        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False, 
                 _anim: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
