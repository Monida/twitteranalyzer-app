"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var logging_1 = require("../../core/logging");
var dom_1 = require("../../core/dom");
var build_views_1 = require("../../core/build_views");
var p = require("../../core/properties");
var dom_view_1 = require("../../core/dom_view");
var array_1 = require("../../core/util/array");
var data_structures_1 = require("../../core/util/data_structures");
var types_1 = require("../../core/util/types");
var model_1 = require("../../model");
var gesture_tool_1 = require("./gestures/gesture_tool");
var action_tool_1 = require("./actions/action_tool");
var help_tool_1 = require("./actions/help_tool");
var inspect_tool_1 = require("./inspectors/inspect_tool");
var toolbar_1 = require("../../styles/toolbar");
var logo_1 = require("../../styles/logo");
var mixins_1 = require("../../styles/mixins");
var ToolbarViewModel = /** @class */ (function (_super) {
    tslib_1.__extends(ToolbarViewModel, _super);
    function ToolbarViewModel(attrs) {
        return _super.call(this, attrs) || this;
    }
    ToolbarViewModel.initClass = function () {
        this.define({
            _visible: [p.Any, null],
            autohide: [p.Boolean, false],
        });
    };
    Object.defineProperty(ToolbarViewModel.prototype, "visible", {
        get: function () {
            return (!this.autohide) ? true : (this._visible == null) ? false : this._visible;
        },
        enumerable: true,
        configurable: true
    });
    ToolbarViewModel.__name__ = "ToolbarViewModel";
    return ToolbarViewModel;
}(model_1.Model));
exports.ToolbarViewModel = ToolbarViewModel;
ToolbarViewModel.initClass();
var ToolbarBaseView = /** @class */ (function (_super) {
    tslib_1.__extends(ToolbarBaseView, _super);
    function ToolbarBaseView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ToolbarBaseView.prototype.initialize = function () {
        _super.prototype.initialize.call(this);
        this._tool_button_views = {};
        this._build_tool_button_views();
        this._toolbar_view_model = new ToolbarViewModel({ autohide: this.model.autohide });
    };
    ToolbarBaseView.prototype.connect_signals = function () {
        var _this = this;
        _super.prototype.connect_signals.call(this);
        this.connect(this.model.properties.tools.change, function () {
            _this._build_tool_button_views();
            _this.render();
        });
        this.connect(this.model.properties.autohide.change, function () {
            _this._toolbar_view_model.autohide = _this.model.autohide;
            _this._on_visible_change();
        });
        this.connect(this._toolbar_view_model.properties._visible.change, function () { return _this._on_visible_change(); });
    };
    ToolbarBaseView.prototype.remove = function () {
        build_views_1.remove_views(this._tool_button_views);
        _super.prototype.remove.call(this);
    };
    ToolbarBaseView.prototype._build_tool_button_views = function () {
        var tools = (this.model._proxied_tools != null ? this.model._proxied_tools : this.model.tools); // XXX
        build_views_1.build_views(this._tool_button_views, tools, { parent: this }, function (tool) { return tool.button_view; });
    };
    ToolbarBaseView.prototype.set_visibility = function (visible) {
        if (visible != this._toolbar_view_model._visible) {
            this._toolbar_view_model._visible = visible;
        }
    };
    ToolbarBaseView.prototype._on_visible_change = function () {
        var visible = this._toolbar_view_model.visible;
        var hidden_class = toolbar_1.bk_toolbar_hidden;
        if (this.el.classList.contains(hidden_class) && visible) {
            this.el.classList.remove(hidden_class);
        }
        else if (!visible) {
            this.el.classList.add(hidden_class);
        }
    };
    ToolbarBaseView.prototype.render = function () {
        var _this = this;
        dom_1.empty(this.el);
        this.el.classList.add(toolbar_1.bk_toolbar);
        this.el.classList.add(mixins_1.bk_side(this.model.toolbar_location));
        this._toolbar_view_model.autohide = this.model.autohide;
        this._on_visible_change();
        if (this.model.logo != null) {
            var gray = this.model.logo === "grey" ? logo_1.bk_grey : null;
            var logo = dom_1.a({ href: "https://bokeh.pydata.org/", target: "_blank", class: [logo_1.bk_logo, logo_1.bk_logo_small, gray] });
            this.el.appendChild(logo);
        }
        var bars = [];
        var el = function (tool) {
            return _this._tool_button_views[tool.id].el;
        };
        var gestures = this.model.gestures;
        for (var et in gestures) {
            bars.push(gestures[et].tools.map(el));
        }
        bars.push(this.model.actions.map(el));
        bars.push(this.model.inspectors.filter(function (tool) { return tool.toggleable; }).map(el));
        bars.push(this.model.help.map(el));
        for (var _i = 0, bars_1 = bars; _i < bars_1.length; _i++) {
            var bar = bars_1[_i];
            if (bar.length !== 0) {
                var el_1 = dom_1.div({ class: toolbar_1.bk_button_bar }, bar);
                this.el.appendChild(el_1);
            }
        }
    };
    ToolbarBaseView.prototype.update_layout = function () { };
    ToolbarBaseView.prototype.update_position = function () { };
    ToolbarBaseView.prototype.after_layout = function () {
        this._has_finished = true;
    };
    ToolbarBaseView.__name__ = "ToolbarBaseView";
    return ToolbarBaseView;
}(dom_view_1.DOMView));
exports.ToolbarBaseView = ToolbarBaseView;
function createGestureMap() {
    return {
        pan: { tools: [], active: null },
        scroll: { tools: [], active: null },
        pinch: { tools: [], active: null },
        tap: { tools: [], active: null },
        doubletap: { tools: [], active: null },
        press: { tools: [], active: null },
        rotate: { tools: [], active: null },
        move: { tools: [], active: null },
        multi: { tools: [], active: null },
    };
}
var ToolbarBase = /** @class */ (function (_super) {
    tslib_1.__extends(ToolbarBase, _super);
    function ToolbarBase(attrs) {
        return _super.call(this, attrs) || this;
    }
    ToolbarBase.initClass = function () {
        this.prototype.default_view = ToolbarBaseView;
        this.define({
            tools: [p.Array, []],
            logo: [p.Logo, 'normal'],
            autohide: [p.Boolean, false],
        });
        this.internal({
            gestures: [p.Any, createGestureMap],
            actions: [p.Array, []],
            inspectors: [p.Array, []],
            help: [p.Array, []],
            toolbar_location: [p.Location, 'right'],
        });
    };
    ToolbarBase.prototype.initialize = function () {
        _super.prototype.initialize.call(this);
        this._init_tools();
    };
    ToolbarBase.prototype._init_tools = function () {
        var _this = this;
        // The only purpose of this function is to avoid unnecessary property churning.
        var tools_changed = function (old_tools, new_tools) {
            if (old_tools.length != new_tools.length) {
                return true;
            }
            var new_ids = new data_structures_1.Set(new_tools.map(function (t) { return t.id; }));
            return array_1.some(old_tools, function (t) { return !new_ids.has(t.id); });
        };
        var new_inspectors = this.tools.filter(function (t) { return t instanceof inspect_tool_1.InspectTool; });
        if (tools_changed(this.inspectors, new_inspectors)) {
            this.inspectors = new_inspectors;
        }
        var new_help = this.tools.filter(function (t) { return t instanceof help_tool_1.HelpTool; });
        if (tools_changed(this.help, new_help)) {
            this.help = new_help;
        }
        var new_actions = this.tools.filter(function (t) { return t instanceof action_tool_1.ActionTool; });
        if (tools_changed(this.actions, new_actions)) {
            this.actions = new_actions;
        }
        var check_event_type = function (et, tool) {
            if (!(et in _this.gestures)) {
                logging_1.logger.warn("Toolbar: unknown event type '" + et + "' for tool: " + tool.type + " (" + tool.id + ")");
            }
        };
        var new_gestures = createGestureMap();
        for (var _i = 0, _a = this.tools; _i < _a.length; _i++) {
            var tool = _a[_i];
            if (tool instanceof gesture_tool_1.GestureTool && tool.event_type) {
                if (types_1.isString(tool.event_type)) {
                    new_gestures[tool.event_type].tools.push(tool);
                    check_event_type(tool.event_type, tool);
                }
                else {
                    new_gestures.multi.tools.push(tool);
                    for (var _b = 0, _c = tool.event_type; _b < _c.length; _b++) {
                        var et = _c[_b];
                        check_event_type(et, tool);
                    }
                }
            }
        }
        var _loop_1 = function (et) {
            var gm = this_1.gestures[et];
            if (tools_changed(gm.tools, new_gestures[et].tools)) {
                gm.tools = new_gestures[et].tools;
            }
            if (gm.active && array_1.every(gm.tools, function (t) { return t.id != gm.active.id; })) {
                gm.active = null;
            }
        };
        var this_1 = this;
        for (var _d = 0, _e = Object.keys(new_gestures); _d < _e.length; _d++) {
            var et = _e[_d];
            _loop_1(et);
        }
    };
    Object.defineProperty(ToolbarBase.prototype, "horizontal", {
        get: function () {
            return this.toolbar_location === "above" || this.toolbar_location === "below";
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ToolbarBase.prototype, "vertical", {
        get: function () {
            return this.toolbar_location === "left" || this.toolbar_location === "right";
        },
        enumerable: true,
        configurable: true
    });
    ToolbarBase.prototype._active_change = function (tool) {
        var event_type = tool.event_type;
        if (event_type == null)
            return;
        var event_types = types_1.isString(event_type) ? [event_type] : event_type;
        for (var _i = 0, event_types_1 = event_types; _i < event_types_1.length; _i++) {
            var et = event_types_1[_i];
            if (tool.active) {
                var currently_active_tool = this.gestures[et].active;
                if (currently_active_tool != null && tool != currently_active_tool) {
                    logging_1.logger.debug("Toolbar: deactivating tool: " + currently_active_tool.type + " (" + currently_active_tool.id + ") for event type '" + et + "'");
                    currently_active_tool.active = false;
                }
                this.gestures[et].active = tool;
                logging_1.logger.debug("Toolbar: activating tool: " + tool.type + " (" + tool.id + ") for event type '" + et + "'");
            }
            else
                this.gestures[et].active = null;
        }
    };
    ToolbarBase.__name__ = "ToolbarBase";
    return ToolbarBase;
}(model_1.Model));
exports.ToolbarBase = ToolbarBase;
ToolbarBase.initClass();
