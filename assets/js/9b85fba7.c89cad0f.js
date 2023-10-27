"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6676],{3905:(e,t,r)=>{r.d(t,{Zo:()=>f,kt:()=>m});var n=r(7294);function l(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){l(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function o(e,t){if(null==e)return{};var r,n,l=function(e,t){if(null==e)return{};var r,n,l={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(l[r]=e[r]);return l}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(l[r]=e[r])}return l}var d=n.createContext({}),s=function(e){var t=n.useContext(d),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},f=function(e){var t=s(e.components);return n.createElement(d.Provider,{value:t},e.children)},c="mdxType",p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var r=e.components,l=e.mdxType,a=e.originalType,d=e.parentName,f=o(e,["components","mdxType","originalType","parentName"]),c=s(r),u=l,m=c["".concat(d,".").concat(u)]||c[u]||p[u]||a;return r?n.createElement(m,i(i({ref:t},f),{},{components:r})):n.createElement(m,i({ref:t},f))}));function m(e,t){var r=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var a=r.length,i=new Array(a);i[0]=u;var o={};for(var d in t)hasOwnProperty.call(t,d)&&(o[d]=t[d]);o.originalType=e,o[c]="string"==typeof e?e:l,i[1]=o;for(var s=2;s<a;s++)i[s]=r[s];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}u.displayName="MDXCreateElement"},8992:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>d,contentTitle:()=>i,default:()=>p,frontMatter:()=>a,metadata:()=>o,toc:()=>s});var n=r(7462),l=(r(7294),r(3905));const a={id:"fields.layer_fields_3d"},i=void 0,o={unversionedId:"API/fields.layer_fields_3d",id:"API/fields.layer_fields_3d",title:"fields.layer_fields_3d",description:"fields.layerfields3d",source:"@site/docs/API/fields.layer_fields_3d.md",sourceDirName:"API",slug:"/API/fields.layer_fields_3d",permalink:"/fmmax/API/fields.layer_fields_3d",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fields.layer_fields_3d.md",tags:[],version:"current",frontMatter:{id:"fields.layer_fields_3d"},sidebar:"APISidebar",previous:{title:"fields.fields_on_grid",permalink:"/fmmax/API/fields.fields_on_grid"},next:{title:"fields.layer_fields_3d_on_coordinates",permalink:"/fmmax/API/fields.layer_fields_3d_on_coordinates"}},d={},s=[{value:"<code>fields.layer_fields_3d</code>",id:"fieldslayer_fields_3d",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],f={toc:s},c="wrapper";function p(e){let{components:t,...r}=e;return(0,l.kt)(c,(0,n.Z)({},f,r,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h3",{id:"fieldslayer_fields_3d"},(0,l.kt)("inlineCode",{parentName:"h3"},"fields.layer_fields_3d")),(0,l.kt)("p",null,"Computes the three-dimensional fields in a layer on the real-space grid."),(0,l.kt)("h4",{id:"args"},"Args:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"forward_amplitude_start"),": The forward-going wave amplitudes, defined at the\nstart of the layer."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"backward_amplitude_end"),": The backward-going wave amplitudes, defined at the\nend of the layer."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"layer_solve_result"),": The results of the layer eigensolve."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"layer_thickness"),": The layer thickness."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"layer_znum"),": The number of gridpoints in the z-direction for the layer."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"grid_shape"),": The shape of the xy real-space grid."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"num_unit_cells"),": The number of unit cells along each direction.")),(0,l.kt)("h4",{id:"returns"},"Returns:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"None"),": The electric and magnetic fields and grid coordinates, ",(0,l.kt)("inlineCode",{parentName:"li"},"(ef, hf, (x, y, z))"),".")))}p.isMDXComponent=!0}}]);