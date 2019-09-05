"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var p = require("../../core/properties");
var widget_1 = require("./widget");
var FileInputView = /** @class */ (function (_super) {
    tslib_1.__extends(FileInputView, _super);
    function FileInputView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    FileInputView.prototype.connect_signals = function () {
        var _this = this;
        _super.prototype.connect_signals.call(this);
        this.connect(this.model.change, function () { return _this.render(); });
        this.connect(this.model.properties.width.change, function () { return _this.render(); });
    };
    FileInputView.prototype.render = function () {
        var _this = this;
        if (this.dialogEl) {
            return;
        }
        this.dialogEl = document.createElement('input');
        this.dialogEl.type = "file";
        this.dialogEl.multiple = false;
        if (this.model.accept != null && this.model.accept != '')
            this.dialogEl.accept = this.model.accept;
        this.dialogEl.style.width = "{this.model.width}px";
        this.dialogEl.onchange = function (e) { return _this.load_file(e); };
        this.el.appendChild(this.dialogEl);
    };
    FileInputView.prototype.load_file = function (e) {
        var _this = this;
        var reader = new FileReader();
        this.model.filename = e.target.files[0].name;
        reader.onload = function (e) { return _this.file(e); };
        reader.readAsDataURL(e.target.files[0]);
    };
    FileInputView.prototype.file = function (e) {
        var file = e.target.result;
        var file_arr = file.split(",");
        var content = file_arr[1];
        var header = file_arr[0].split(":")[1].split(";")[0];
        this.model.value = content;
        this.model.mime_type = header;
    };
    FileInputView.__name__ = "FileInputView";
    return FileInputView;
}(widget_1.WidgetView));
exports.FileInputView = FileInputView;
var FileInput = /** @class */ (function (_super) {
    tslib_1.__extends(FileInput, _super);
    function FileInput(attrs) {
        return _super.call(this, attrs) || this;
    }
    FileInput.initClass = function () {
        this.prototype.type = "FileInput";
        this.prototype.default_view = FileInputView;
        this.define({
            value: [p.String, ''],
            mime_type: [p.String, ''],
            filename: [p.String, ''],
            accept: [p.String, ''],
        });
    };
    FileInput.__name__ = "FileInput";
    return FileInput;
}(widget_1.Widget));
exports.FileInput = FileInput;
FileInput.initClass();
