#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-face-censor", "en-US")

void register_face_censor();

bool obs_module_load(void)
{
	register_face_censor();
	return true;
}
