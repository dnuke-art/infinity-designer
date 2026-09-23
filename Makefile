# infinity-designer
#
#   make dev                     run the app at http://localhost:5173
#   make build                   type-check and build dist/
#   make render SCENE=scenes/x.json [OUT=renders/x.png] [SAMPLES=512] [RES=1920x1080] [EXPOSURE=-2] [JPG=92]
#   make renders                 render every scenes/*.json to renders/*.png
#   make renders/x.jpg           a JPEG of scenes/x.json
#   make blend SCENE=scenes/x.json   open the scene in Blender's window, rendered viewport
#   make blends/x.blend          save scenes/x.json as a .blend to reopen later
#
# Scenes come from the app's "export for Blender" button: save the JSON into scenes/.
# Headless Blender needs the C locale (an OCIO locale bug segfaults it otherwise) and, on
# this machine, Blender 5.0.1: 5.2 segfaults in its oneAPI device probe.

SHELL := /bin/bash
SAMPLES ?= 512
RES ?= 1920x1080
EXPOSURE ?=
JPG ?= 92
BLENDER ?= $(firstword $(wildcard /opt/blender-5.0.1-linux-x64/blender) $(shell command -v blender 2>/dev/null))
RES_ARGS := $(subst x, ,$(RES))
EXPO_ARGS := $(if $(EXPOSURE),--exposure $(EXPOSURE),)
SCENES := $(wildcard scenes/*.json)
PNGS := $(patsubst scenes/%.json,renders/%.png,$(SCENES))

.PHONY: dev build check preview install render renders blend clean help

help:
	@sed -n '2,12p' Makefile | sed 's/^# \{0,1\}//'

install: node_modules
node_modules: package.json
	npm install
	@touch node_modules

dev: node_modules
	npm run dev

build: node_modules
	npm run build

check: node_modules
	npx tsc --noEmit

preview: build
	npm run preview

OUT ?= renders/$(basename $(notdir $(SCENE))).png
BLENDER_CMD = LC_ALL=C LANG=C $(BLENDER) -b -P tools/blender_render.py

render:
	@test -n "$(SCENE)" || { echo "usage: make render SCENE=scenes/name.json [OUT=renders/name.png]"; exit 1; }
	@test -n "$(BLENDER)" || { echo "no blender found; set BLENDER=/path/to/blender"; exit 1; }
	@mkdir -p $(dir $(OUT))
	$(BLENDER_CMD) -- $(SCENE) $(OUT) --samples $(SAMPLES) --res $(RES_ARGS) $(EXPO_ARGS) $(if $(filter %.jpg %.jpeg,$(OUT)),--jpg $(JPG),) 2>&1 | tail -3
	@test -s $(OUT) && echo "wrote $(OUT)" || { echo "render failed: $(OUT) missing"; exit 1; }

renders: $(PNGS)

# interactive: Blender's window with the scene built and a Cycles rendered viewport
blend:
	@test -n "$(SCENE)" || { echo "usage: make blend SCENE=scenes/name.json"; exit 1; }
	@test -n "$(BLENDER)" || { echo "no blender found; set BLENDER=/path/to/blender"; exit 1; }
	LC_ALL=C LANG=C $(BLENDER) -P tools/blender_render.py -- $(SCENE) --interactive $(EXPO_ARGS) &

blends/%.blend: scenes/%.json tools/blender_render.py
	@test -n "$(BLENDER)" || { echo "no blender found; set BLENDER=/path/to/blender"; exit 1; }
	@mkdir -p blends
	$(BLENDER_CMD) -- $< --blend $@ $(EXPO_ARGS) 2>&1 | tail -2
	@test -s $@ && echo "saved $@" || { echo "save failed: $@ missing"; exit 1; }

renders/%.png: scenes/%.json tools/blender_render.py
	@test -n "$(BLENDER)" || { echo "no blender found; set BLENDER=/path/to/blender"; exit 1; }
	@mkdir -p renders
	$(BLENDER_CMD) -- $< $@ --samples $(SAMPLES) --res $(RES_ARGS) $(EXPO_ARGS) 2>&1 | tail -3
	@test -s $@ && echo "wrote $@" || { echo "render failed: $@ missing"; exit 1; }

renders/%.jpg: scenes/%.json tools/blender_render.py
	@test -n "$(BLENDER)" || { echo "no blender found; set BLENDER=/path/to/blender"; exit 1; }
	@mkdir -p renders
	$(BLENDER_CMD) -- $< $@ --samples $(SAMPLES) --res $(RES_ARGS) $(EXPO_ARGS) --jpg $(JPG) 2>&1 | tail -3
	@test -s $@ && echo "wrote $@" || { echo "render failed: $@ missing"; exit 1; }

clean:
	rm -rf dist renders/*.png renders/*.jpg blends/*.blend
