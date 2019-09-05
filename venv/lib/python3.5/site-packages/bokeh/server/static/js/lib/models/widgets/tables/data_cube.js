"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var p = require("../../../core/properties");
var dom_1 = require("../../../core/dom");
var slickgrid_1 = require("slickgrid");
var data_table_1 = require("./data_table");
var model_1 = require("../../../model");
function groupCellFormatter(_row, _cell, _value, _columnDef, dataContext) {
    var collapsed = dataContext.collapsed, level = dataContext.level, title = dataContext.title;
    var toggle = dom_1.span({
        class: "slick-group-toggle " + (collapsed ? 'collapsed' : 'expanded'),
        style: { 'margin-left': level * 15 + "px" },
    });
    var titleElement = dom_1.span({
        class: 'slick-group-title',
        level: level,
    }, title);
    return "" + toggle.outerHTML + titleElement.outerHTML;
}
function indentFormatter(formatter, indent) {
    return function (row, cell, value, columnDef, dataContext) {
        var spacer = dom_1.span({
            class: 'slick-group-toggle',
            style: { 'margin-left': (indent || 0) * 15 + "px" },
        });
        var formatted = formatter ? formatter(row, cell, value, columnDef, dataContext) : "" + value;
        return "" + spacer.outerHTML + (formatted && formatted.replace(/^<div/, '<span').replace(/div>$/, 'span>'));
    };
}
function handleGridClick(event, args) {
    var item = this.getDataItem(args.row);
    if (item instanceof slickgrid_1.Group && event.target.classList.contains('slick-group-toggle')) {
        if (item.collapsed) {
            this.getData().expandGroup(item.groupingKey);
        }
        else {
            this.getData().collapseGroup(item.groupingKey);
        }
        event.stopImmediatePropagation();
        event.preventDefault();
        this.invalidate();
        this.render();
    }
}
var GroupingInfo = /** @class */ (function (_super) {
    tslib_1.__extends(GroupingInfo, _super);
    function GroupingInfo(attrs) {
        return _super.call(this, attrs) || this;
    }
    GroupingInfo.initClass = function () {
        this.prototype.type = 'GroupingInfo';
        this.define({
            getter: [p.String, ''],
            aggregators: [p.Array, []],
            collapsed: [p.Boolean, false],
        });
    };
    Object.defineProperty(GroupingInfo.prototype, "comparer", {
        get: function () {
            return function (a, b) {
                return a.value === b.value ? 0 : a.value > b.value ? 1 : -1;
            };
        },
        enumerable: true,
        configurable: true
    });
    GroupingInfo.__name__ = "GroupingInfo";
    return GroupingInfo;
}(model_1.Model));
exports.GroupingInfo = GroupingInfo;
GroupingInfo.initClass();
var DataCubeProvider = /** @class */ (function (_super) {
    tslib_1.__extends(DataCubeProvider, _super);
    function DataCubeProvider(source, view, columns, target) {
        var _this = _super.call(this, source, view) || this;
        _this.columns = columns;
        _this.groupingInfos = [];
        _this.groupingDelimiter = ':|:';
        _this.target = target;
        return _this;
    }
    DataCubeProvider.prototype.setGrouping = function (groupingInfos) {
        this.groupingInfos = groupingInfos;
        this.toggledGroupsByLevel = groupingInfos.map(function () { return ({}); });
        this.refresh();
    };
    DataCubeProvider.prototype.extractGroups = function (rows, parentGroup) {
        var _this = this;
        var groups = [];
        var groupsByValue = new Map();
        var level = parentGroup ? parentGroup.level + 1 : 0;
        var _a = this.groupingInfos[level], comparer = _a.comparer, getter = _a.getter;
        rows.forEach(function (row) {
            var value = _this.source.data[getter][row];
            var group = groupsByValue.get(value);
            if (!group) {
                var groupingKey = parentGroup ? "" + parentGroup.groupingKey + _this.groupingDelimiter + value : "" + value;
                group = Object.assign(new slickgrid_1.Group(), { value: value, level: level, groupingKey: groupingKey });
                groups.push(group);
                groupsByValue.set(value, group);
            }
            group.rows.push(row);
        });
        if (level < this.groupingInfos.length - 1) {
            groups.forEach(function (group) {
                group.groups = _this.extractGroups(group.rows, group);
            });
        }
        groups.sort(comparer);
        return groups;
    };
    DataCubeProvider.prototype.calculateTotals = function (group, aggregators) {
        var totals = { avg: {}, max: {}, min: {}, sum: {} };
        var data = this.source.data;
        var keys = Object.keys(data);
        var items = group.rows.map(function (i) { return keys.reduce(function (o, c) {
            var _a;
            return (tslib_1.__assign({}, o, (_a = {}, _a[c] = data[c][i], _a)));
        }, {}); });
        aggregators.forEach(function (aggregator) {
            aggregator.init();
            items.forEach(function (item) { return aggregator.accumulate(item); });
            aggregator.storeResult(totals);
        });
        return totals;
    };
    DataCubeProvider.prototype.addTotals = function (groups, level) {
        var _this = this;
        if (level === void 0) { level = 0; }
        var _a = this.groupingInfos[level], aggregators = _a.aggregators, groupCollapsed = _a.collapsed;
        var toggledGroups = this.toggledGroupsByLevel[level];
        groups.forEach(function (group) {
            if (group.groups) {
                _this.addTotals(group.groups, level + 1);
            }
            if (aggregators.length && group.rows.length) {
                group.totals = _this.calculateTotals(group, aggregators);
            }
            group.collapsed = groupCollapsed !== toggledGroups[group.groupingKey];
            group.title = group.value ? "" + group.value : "";
        });
    };
    DataCubeProvider.prototype.flattenedGroupedRows = function (groups, level) {
        var _this = this;
        if (level === void 0) { level = 0; }
        var rows = [];
        groups.forEach(function (group) {
            rows.push(group);
            if (!group.collapsed) {
                var subRows = group.groups
                    ? _this.flattenedGroupedRows(group.groups, level + 1)
                    : group.rows;
                rows.push.apply(rows, subRows);
            }
        });
        return rows;
    };
    DataCubeProvider.prototype.refresh = function () {
        var groups = this.extractGroups(this.view.indices);
        var labels = this.source.data[this.columns[0].field];
        if (groups.length) {
            this.addTotals(groups);
            this.rows = this.flattenedGroupedRows(groups);
            this.target.data = {
                row_indices: this.rows.map(function (value) { return value instanceof slickgrid_1.Group ? value.rows : value; }),
                labels: this.rows.map(function (value) { return value instanceof slickgrid_1.Group ? value.title : labels[value]; }),
            };
        }
    };
    DataCubeProvider.prototype.getLength = function () {
        return this.rows.length;
    };
    DataCubeProvider.prototype.getItem = function (i) {
        var _a;
        var item = this.rows[i];
        var data = this.source.data;
        return item instanceof slickgrid_1.Group
            ? item
            : Object.keys(data)
                .reduce(function (o, c) {
                var _a;
                return (tslib_1.__assign({}, o, (_a = {}, _a[c] = data[c][item], _a)));
            }, (_a = {}, _a[data_table_1.DTINDEX_NAME] = item, _a));
    };
    DataCubeProvider.prototype.getItemMetadata = function (i) {
        var myItem = this.rows[i];
        var columns = this.columns.slice(1);
        var aggregators = myItem instanceof slickgrid_1.Group
            ? this.groupingInfos[myItem.level].aggregators
            : [];
        function adapter(column) {
            var myField = column.field, formatter = column.formatter;
            var aggregator = aggregators.find(function (_a) {
                var field_ = _a.field_;
                return field_ === myField;
            });
            if (aggregator) {
                var key_1 = aggregator.key;
                return {
                    formatter: function (row, cell, _value, columnDef, dataContext) {
                        return formatter ? formatter(row, cell, dataContext.totals[key_1][myField], columnDef, dataContext) : '';
                    },
                };
            }
            return {};
        }
        return myItem instanceof slickgrid_1.Group
            ? {
                selectable: false,
                focusable: false,
                cssClasses: 'slick-group',
                columns: [{ formatter: groupCellFormatter }].concat(columns.map(adapter)),
            }
            : {};
    };
    DataCubeProvider.prototype.collapseGroup = function (groupingKey) {
        var level = groupingKey.split(this.groupingDelimiter).length - 1;
        this.toggledGroupsByLevel[level][groupingKey] = !this.groupingInfos[level].collapsed;
        this.refresh();
    };
    DataCubeProvider.prototype.expandGroup = function (groupingKey) {
        var level = groupingKey.split(this.groupingDelimiter).length - 1;
        this.toggledGroupsByLevel[level][groupingKey] = this.groupingInfos[level].collapsed;
        this.refresh();
    };
    DataCubeProvider.__name__ = "DataCubeProvider";
    return DataCubeProvider;
}(data_table_1.TableDataProvider));
exports.DataCubeProvider = DataCubeProvider;
var DataCubeView = /** @class */ (function (_super) {
    tslib_1.__extends(DataCubeView, _super);
    function DataCubeView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    DataCubeView.prototype.render = function () {
        var options = {
            enableCellNavigation: this.model.selectable !== false,
            enableColumnReorder: false,
            forceFitColumns: this.model.fit_columns,
            multiColumnSort: false,
            editable: this.model.editable,
            autoEdit: false,
            rowHeight: this.model.row_height,
        };
        var columns = this.model.columns.map(function (column) { return column.toColumn(); });
        columns[0].formatter = indentFormatter(columns[0].formatter, this.model.grouping.length);
        delete columns[0].editor;
        this.data = new DataCubeProvider(this.model.source, this.model.view, columns, this.model.target);
        this.data.setGrouping(this.model.grouping);
        this.el.style.width = this.model.width + "px";
        this.grid = new slickgrid_1.Grid(this.el, this.data, columns, options);
        this.grid.onClick.subscribe(handleGridClick);
    };
    DataCubeView.__name__ = "DataCubeView";
    return DataCubeView;
}(data_table_1.DataTableView));
exports.DataCubeView = DataCubeView;
var DataCube = /** @class */ (function (_super) {
    tslib_1.__extends(DataCube, _super);
    function DataCube(attrs) {
        return _super.call(this, attrs) || this;
    }
    DataCube.initClass = function () {
        this.prototype.type = 'DataCube';
        this.prototype.default_view = DataCubeView;
        this.define({
            grouping: [p.Array, []],
            target: [p.Instance],
        });
    };
    DataCube.__name__ = "DataCube";
    return DataCube;
}(data_table_1.DataTable));
exports.DataCube = DataCube;
DataCube.initClass();
