"""CPU-only checks for endpoint geometry; no model loading or live queue jobs."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
pkg = types.ModuleType('star7_endpoint_crop_tests')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
module = importlib.import_module(pkg.__name__ + '.h3_face_refine_star7')


class Tests(unittest.TestCase):
    def test_center_region_is_retained_in_both_orientations(self):
        for height, width in ((64, 128), (128, 64)):
            image = torch.zeros(1, height, width, 3)
            image[:, height//2-8:height//2+8, width//2-8:width//2+8] = 1
            result = module._fit_endpoint_image(image, 64, 64)
            expected = image[:, (height-64)//2:(height+64)//2, (width-64)//2:(width+64)//2]
            self.assertTrue(torch.equal(result, expected))

    def test_scaled_square_stays_square_in_landscape_canvas(self):
        image = torch.zeros(1, 160, 160, 3)
        image[:, 64:96, 64:96] = 1
        result = module._fit_endpoint_image(image, 128, 64)
        y, x = torch.where(result[0, :, :, 0] > .5)
        self.assertLessEqual(abs(int(x.max()-x.min()) - int(y.max()-y.min())), 1)
        self.assertEqual(tuple(result.shape), (1, 64, 128, 3))

    def test_missing_and_matching_endpoints_do_not_copy(self):
        self.assertIsNone(module._fit_endpoint_image(None, 128, 64))
        image = torch.zeros(1, 64, 128, 3)
        self.assertIs(module._fit_endpoint_image(image, 128, 64), image)

    def test_first_last_and_paired_modes_fit_inputs_but_keep_original_context(self):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        for use_first, use_last in ((True, False), (False, True), (True, True)):
            first = torch.rand(1, 64, 128, 3) if use_first else None
            last = torch.rand(1, 128, 64, 3) if use_last else None
            with patch.object(MiniMaxH3ImageToVideo, 'execute', return_value=([], {})) as execute:
                result = module.MiniMaxH3MaterialPromptStar7.execute(
                    model=object(), clip=object(), video_vae=object(), audio_vae=object(),
                    prompt='test', width=64, height=64, length=5,
                    first_frame=first, last_frame=last,
                )
            for name, image in (('first_frame', first), ('last_frame', last)):
                fitted = execute.call_args.kwargs[name]
                if image is None:
                    self.assertIsNone(fitted)
                else:
                    self.assertEqual(tuple(fitted.shape), (1, 64, 64, 3))
                self.assertIs(result.result[4][name], image)


if __name__ == '__main__':
    unittest.main()
