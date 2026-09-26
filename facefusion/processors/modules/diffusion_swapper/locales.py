from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the diffusion model responsible for regenerating the face',
			'strength': 'specify how much the face should be regenerated, higher values deviate more from the target',
			'steps': 'specify the number of denoising steps',
			'scale': 'specify the guidance scale of the diffusion process',
			'seed': 'specify a fixed seed for reproducible results'
		},
		'uis':
		{
			'model_dropdown': 'DIFFUSION SWAPPER MODEL',
			'strength_slider': 'DIFFUSION SWAPPER STRENGTH',
			'steps_slider': 'DIFFUSION SWAPPER STEPS',
			'scale_slider': 'DIFFUSION SWAPPER GUIDANCE SCALE',
			'seed_number': 'DIFFUSION SWAPPER SEED'
		}
	}
}
