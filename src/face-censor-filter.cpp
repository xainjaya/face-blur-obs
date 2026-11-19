// Face Censor - OBS video filter
// Detects faces (OpenCV YuNet) and censors them on the GPU: blur / black box / static / pixelate.

#include <obs-module.h>
#include <graphics/graphics.h>
#include <graphics/vec4.h>
#include <util/platform.h>
#include <util/bmem.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define MAX_FACES 8
#define DETECT_WIDTH 640 // frames are downscaled to this width before detection

enum Mode { MODE_BLUR = 0, MODE_BLACK = 1, MODE_STATIC = 2, MODE_PIXELATE = 3 };

struct Rect {
	float x, y, w, h; // normalized 0..1, top-left origin
};

struct Track {
	Rect r;
	float vx = 0, vy = 0; // normalized units / second
	uint64_t last_seen = 0;
	uint64_t last_upd = 0;
};

struct fc_data {
	obs_source_t *src = nullptr;
	gs_effect_t *effect = nullptr;
	gs_texrender_t *render = nullptr;
	gs_stagesurf_t *stage = nullptr;
	uint32_t stage_w = 0, stage_h = 0;

	// settings
	int mode = MODE_BLUR;
	float padding = 0.35f;
	int blur_radius = 12;
	int pixel_size = 24;
	bool ellipse = false;
	uint64_t hold_ns = 500ull * 1000000ull;
	uint64_t interval_ns = 66ull * 1000000ull;
	float score = 0.6f;
	bool fail_closed = true;
	std::string model_path;

	// detector thread
	std::thread th;
	std::mutex mtx;
	std::condition_variable cv;
	bool quit = false;
	bool pending = false;
	std::atomic<bool> busy{false};
	std::atomic<bool> model_ok{true};
	std::atomic<bool> model_dirty{true};
	std::vector<uint8_t> frame; // BGRA
	int fw = 0, fh = 0;
	std::vector<Rect> results;
	uint32_t result_gen = 0;
	float conf = 0.6f;

	// render-thread state
	uint32_t seen_gen = 0;
	uint64_t last_submit = 0;
	std::vector<Track> tracks;
};

// ---------------------------------------------------------------- detector thread

static void detect_thread(fc_data *d)
{
	cv::Ptr<cv::FaceDetectorYN> det;
	std::vector<uint8_t> local;
	int w = 0, h = 0;
	float score = 0.6f;

	for (;;) {
		{
			std::unique_lock<std::mutex> lk(d->mtx);
			d->cv.wait(lk, [d] { return d->quit || d->pending; });
			if (d->quit)
				return;
			local.swap(d->frame);
			w = d->fw;
			h = d->fh;
			score = d->conf;
			d->pending = false;
		}

		std::vector<Rect> out;
		try {
			if (d->model_dirty.exchange(false) || !det) {
				std::string path;
				{
					std::lock_guard<std::mutex> lk(d->mtx);
					path = d->model_path;
				}
				det = cv::FaceDetectorYN::create(path, "", cv::Size(w, h), score, 0.3f, 50);
				d->model_ok = true;
			}
			det->setInputSize(cv::Size(w, h));
			det->setScoreThreshold(score);

			cv::Mat bgra(h, w, CV_8UC4, local.data());
			cv::Mat bgr;
			cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

			cv::Mat faces;
			det->detect(bgr, faces);
			for (int i = 0; i < faces.rows && (int)out.size() < MAX_FACES; i++) {
				Rect r;
				r.x = faces.at<float>(i, 0) / w;
				r.y = faces.at<float>(i, 1) / h;
				r.w = faces.at<float>(i, 2) / w;
				r.h = faces.at<float>(i, 3) / h;
				out.push_back(r);
			}
		} catch (const cv::Exception &e) {
			if (d->model_ok.exchange(false))
				blog(LOG_WARNING, "[face-censor] detector failed: %s", e.what());
			det.release();
		}

		{
			std::lock_guard<std::mutex> lk(d->mtx);
			d->results = std::move(out);
			d->result_gen++;
		}
		d->busy = false;
	}
}

// ---------------------------------------------------------------- tracking

static float center_dist(const Rect &a, const Rect &b)
{
	float dx = (a.x + a.w / 2) - (b.x + b.w / 2);
	float dy = (a.y + a.h / 2) - (b.y + b.h / 2);
	return std::sqrt(dx * dx + dy * dy);
}

static void update_tracks(fc_data *d, const std::vector<Rect> &dets, uint64_t now)
{
	std::vector<bool> used(d->tracks.size(), false);
	for (const Rect &r : dets) {
		int best = -1;
		float bd = 1e9f;
		for (size_t i = 0; i < d->tracks.size(); i++) {
			if (used[i])
				continue;
			float dist = center_dist(r, d->tracks[i].r);
			if (dist < bd) {
				bd = dist;
				best = (int)i;
			}
		}
		if (best >= 0 && bd < std::max(r.w, r.h) * 1.5f) {
			Track &t = d->tracks[best];
			used[best] = true;
			float dt = std::max(0.016f, (float)(now - t.last_upd) / 1e9f);
			Rect n;
			const float a = 0.6f; // smoothing toward new detection
			n.x = t.r.x + (r.x - t.r.x) * a;
			n.y = t.r.y + (r.y - t.r.y) * a;
			n.w = t.r.w + (r.w - t.r.w) * a;
			n.h = t.r.h + (r.h - t.r.h) * a;
			t.vx = 0.5f * t.vx + 0.5f * (n.x - t.r.x) / dt;
			t.vy = 0.5f * t.vy + 0.5f * (n.y - t.r.y) / dt;
			t.r = n;
			t.last_seen = t.last_upd = now;
		} else if ((int)d->tracks.size() < MAX_FACES) {
			Track t;
			t.r = r;
			t.last_seen = t.last_upd = now;
			d->tracks.push_back(t);
			used.push_back(true);
		}
	}
	d->tracks.erase(std::remove_if(d->tracks.begin(), d->tracks.end(),
				       [&](const Track &t) { return now - t.last_seen > d->hold_ns; }),
			d->tracks.end());
}

// ---------------------------------------------------------------- source callbacks

static const char *fc_name(void *)
{
	return obs_module_text("FaceCensor");
}

static void fc_update(void *data, obs_data_t *s)
{
	fc_data *d = (fc_data *)data;
	d->mode = (int)obs_data_get_int(s, "mode");
	d->padding = (float)obs_data_get_int(s, "padding") / 100.0f;
	d->blur_radius = (int)obs_data_get_int(s, "blur_radius");
	d->pixel_size = (int)obs_data_get_int(s, "pixel_size");
	d->ellipse = obs_data_get_bool(s, "ellipse");
	d->hold_ns = (uint64_t)obs_data_get_int(s, "hold_ms") * 1000000ull;
	d->interval_ns = 1000000000ull / (uint64_t)std::max<int64_t>(1, obs_data_get_int(s, "detect_hz"));
	d->score = (float)obs_data_get_int(s, "score") / 100.0f;
	d->fail_closed = obs_data_get_bool(s, "fail_closed");

	std::string path = obs_data_get_string(s, "model_path");
	std::lock_guard<std::mutex> lk(d->mtx);
	d->conf = d->score;
	if (path != d->model_path) {
		d->model_path = path;
		d->model_dirty = true;
	}
}

static void *fc_create(obs_data_t *settings, obs_source_t *source)
{
	fc_data *d = new fc_data();
	d->src = source;

	char *effect_path = obs_module_file("effects/censor.effect");
	obs_enter_graphics();
	char *err = nullptr;
	d->effect = gs_effect_create_from_file(effect_path, &err);
	d->render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	obs_leave_graphics();
	if (!d->effect)
		blog(LOG_ERROR, "[face-censor] effect load failed: %s", err ? err : "unknown");
	bfree(err);
	bfree(effect_path);

	fc_update(d, settings);
	d->th = std::thread(detect_thread, d);
	return d;
}

static void fc_destroy(void *data)
{
	fc_data *d = (fc_data *)data;
	{
		std::lock_guard<std::mutex> lk(d->mtx);
		d->quit = true;
	}
	d->cv.notify_all();
	if (d->th.joinable())
		d->th.join();

	obs_enter_graphics();
	if (d->stage)
		gs_stagesurface_destroy(d->stage);
	if (d->render)
		gs_texrender_destroy(d->render);
	if (d->effect)
		gs_effect_destroy(d->effect);
	obs_leave_graphics();
	delete d;
}

static void fc_defaults(obs_data_t *s)
{
	char *p = obs_module_file("models/face_detection_yunet_2023mar.onnx");
	obs_data_set_default_string(s, "model_path", p ? p : "");
	bfree(p);
	obs_data_set_default_int(s, "mode", MODE_BLUR);
	obs_data_set_default_int(s, "padding", 35);
	obs_data_set_default_int(s, "blur_radius", 12);
	obs_data_set_default_int(s, "pixel_size", 24);
	obs_data_set_default_bool(s, "ellipse", false);
	obs_data_set_default_int(s, "hold_ms", 500);
	obs_data_set_default_int(s, "detect_hz", 15);
	obs_data_set_default_int(s, "score", 60);
	obs_data_set_default_bool(s, "fail_closed", true);
}

static obs_properties_t *fc_props(void *)
{
	obs_properties_t *p = obs_properties_create();
	obs_property_t *m = obs_properties_add_list(p, "mode", obs_module_text("Mode"), OBS_COMBO_TYPE_LIST,
						    OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(m, obs_module_text("Blur"), MODE_BLUR);
	obs_property_list_add_int(m, obs_module_text("BlackBox"), MODE_BLACK);
	obs_property_list_add_int(m, obs_module_text("Static"), MODE_STATIC);
	obs_property_list_add_int(m, obs_module_text("Pixelate"), MODE_PIXELATE);

	obs_properties_add_int_slider(p, "padding", obs_module_text("Padding"), 0, 100, 1);
	obs_properties_add_int_slider(p, "blur_radius", obs_module_text("BlurStrength"), 1, 40, 1);
	obs_properties_add_int_slider(p, "pixel_size", obs_module_text("PixelSize"), 4, 64, 1);
	obs_properties_add_bool(p, "ellipse", obs_module_text("Ellipse"));
	obs_properties_add_int_slider(p, "hold_ms", obs_module_text("HoldMs"), 0, 2000, 50);
	obs_properties_add_int_slider(p, "detect_hz", obs_module_text("DetectHz"), 5, 30, 1);
	obs_properties_add_int_slider(p, "score", obs_module_text("Confidence"), 10, 95, 1);
	obs_properties_add_bool(p, "fail_closed", obs_module_text("FailClosed"));
	obs_properties_add_path(p, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX (*.onnx)",
				nullptr);
	return p;
}

// Render the source small into a texrender, read it back, and hand it to the detector thread.
static void submit_frame(fc_data *d, obs_source_t *target, uint32_t cx, uint32_t cy)
{
	uint32_t dw = std::min<uint32_t>(DETECT_WIDTH, cx);
	uint32_t dh = std::max<uint32_t>(1, (uint32_t)((uint64_t)cy * dw / cx));

	gs_texrender_reset(d->render);
	if (!gs_texrender_begin(d->render, dw, dh))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)cx, 0.0f, (float)cy, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(d->render);

	gs_texture_t *tex = gs_texrender_get_texture(d->render);
	if (!tex)
		return;
	if (!d->stage || d->stage_w != dw || d->stage_h != dh) {
		if (d->stage)
			gs_stagesurface_destroy(d->stage);
		d->stage = gs_stagesurface_create(dw, dh, GS_BGRA);
		d->stage_w = dw;
		d->stage_h = dh;
	}
	gs_stage_texture(d->stage, tex);

	uint8_t *data = nullptr;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(d->stage, &data, &linesize))
		return;
	{
		std::lock_guard<std::mutex> lk(d->mtx);
		d->frame.resize((size_t)dw * dh * 4);
		for (uint32_t y = 0; y < dh; y++)
			memcpy(&d->frame[(size_t)y * dw * 4], data + (size_t)y * linesize, (size_t)dw * 4);
		d->fw = (int)dw;
		d->fh = (int)dh;
		d->pending = true;
	}
	gs_stagesurface_unmap(d->stage);
	d->busy = true;
	d->cv.notify_one();
}

static void fc_render(void *data, gs_effect_t *)
{
	fc_data *d = (fc_data *)data;
	obs_source_t *target = obs_filter_get_target(d->src);
	uint32_t cx = obs_source_get_base_width(target);
	uint32_t cy = obs_source_get_base_height(target);
	if (!d->effect || !cx || !cy) {
		obs_source_skip_video_filter(d->src);
		return;
	}

	uint64_t now = os_gettime_ns();

	// 1. hand a frame to the detector if it is idle and it's time
	if (!d->busy && now - d->last_submit >= d->interval_ns) {
		d->last_submit = now;
		submit_frame(d, target, cx, cy);
	}

	// 2. pull any new detections into the tracker
	{
		std::vector<Rect> dets;
		bool fresh = false;
		{
			std::lock_guard<std::mutex> lk(d->mtx);
			if (d->result_gen != d->seen_gen) {
				d->seen_gen = d->result_gen;
				dets = d->results;
				fresh = true;
			}
		}
		if (fresh)
			update_tracks(d, dets, now);
		else
			d->tracks.erase(std::remove_if(d->tracks.begin(), d->tracks.end(),
						       [&](const Track &t) { return now - t.last_seen > d->hold_ns; }),
					d->tracks.end());
	}

	// 3. build the rect list (predicted forward, padded, clamped)
	float faces[MAX_FACES * 4] = {0};
	int count = 0;
	int mode = d->mode;
	if (!d->model_ok && d->fail_closed) {
		faces[0] = 0;
		faces[1] = 0;
		faces[2] = 1;
		faces[3] = 1;
		count = 1;
		mode = MODE_BLACK;
	} else {
		for (const Track &t : d->tracks) {
			float dt = std::min(0.15f, (float)(now - t.last_upd) / 1e9f);
			float x = t.r.x + t.vx * dt, y = t.r.y + t.vy * dt;
			float px = t.r.w * d->padding, py = t.r.h * d->padding;
			float x0 = std::max(0.0f, x - px), y0 = std::max(0.0f, y - py);
			float x1 = std::min(1.0f, x + t.r.w + px), y1 = std::min(1.0f, y + t.r.h + py);
			faces[count * 4 + 0] = x0;
			faces[count * 4 + 1] = y0;
			faces[count * 4 + 2] = x1 - x0;
			faces[count * 4 + 3] = y1 - y0;
			if (++count >= MAX_FACES)
				break;
		}
	}

	// 4. draw the source through the censor shader
	if (!obs_source_process_filter_begin(d->src, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
		return;

	gs_eparam_t *p;
	for (int i = 0; i < MAX_FACES; i++) {
		char name[16];
		snprintf(name, sizeof(name), "face%d", i);
		if ((p = gs_effect_get_param_by_name(d->effect, name))) {
			struct vec4 v;
			vec4_set(&v, faces[i * 4], faces[i * 4 + 1], faces[i * 4 + 2], faces[i * 4 + 3]);
			gs_effect_set_vec4(p, &v);
		}
	}
	if ((p = gs_effect_get_param_by_name(d->effect, "count")))
		gs_effect_set_int(p, count);
	if ((p = gs_effect_get_param_by_name(d->effect, "mode")))
		gs_effect_set_int(p, mode);
	if ((p = gs_effect_get_param_by_name(d->effect, "ellipse")))
		gs_effect_set_int(p, d->ellipse ? 1 : 0);
	if ((p = gs_effect_get_param_by_name(d->effect, "radius")))
		gs_effect_set_float(p, (float)d->blur_radius);
	if ((p = gs_effect_get_param_by_name(d->effect, "pixel_size")))
		gs_effect_set_float(p, (float)d->pixel_size);
	if ((p = gs_effect_get_param_by_name(d->effect, "t")))
		gs_effect_set_float(p, (float)std::fmod((double)now / 1e9, 1000.0));
	if ((p = gs_effect_get_param_by_name(d->effect, "texel"))) {
		struct vec2 texel;
		vec2_set(&texel, 1.0f / cx, 1.0f / cy);
		gs_effect_set_vec2(p, &texel);
	}

	obs_source_process_filter_end(d->src, d->effect, cx, cy);
}

struct obs_source_info face_censor_info = {};

void register_face_censor()
{
	face_censor_info.id = "face_censor_filter";
	face_censor_info.type = OBS_SOURCE_TYPE_FILTER;
	face_censor_info.output_flags = OBS_SOURCE_VIDEO;
	face_censor_info.get_name = fc_name;
	face_censor_info.create = fc_create;
	face_censor_info.destroy = fc_destroy;
	face_censor_info.update = fc_update;
	face_censor_info.get_defaults = fc_defaults;
	face_censor_info.get_properties = fc_props;
	face_censor_info.video_render = fc_render;
	obs_register_source(&face_censor_info);
}
