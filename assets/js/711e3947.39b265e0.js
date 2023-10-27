"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2923],{3905:(e,t,r)=>{r.d(t,{Zo:()=>c,kt:()=>m});var n=r(7294);function i(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function l(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?l(Object(r),!0).forEach((function(t){i(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):l(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function o(e,t){if(null==e)return{};var r,n,i=function(e,t){if(null==e)return{};var r,n,i={},l=Object.keys(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||(i[r]=e[r]);return i}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(i[r]=e[r])}return i}var s=n.createContext({}),d=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},c=function(e){var t=d(e.components);return n.createElement(s.Provider,{value:t},e.children)},f="mdxType",u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},p=n.forwardRef((function(e,t){var r=e.components,i=e.mdxType,l=e.originalType,s=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),f=d(r),p=i,m=f["".concat(s,".").concat(p)]||f[p]||u[p]||l;return r?n.createElement(m,a(a({ref:t},c),{},{components:r})):n.createElement(m,a({ref:t},c))}));function m(e,t){var r=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var l=r.length,a=new Array(l);a[0]=p;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[f]="string"==typeof e?e:i,a[1]=o;for(var d=2;d<l;d++)a[d]=r[d];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}p.displayName="MDXCreateElement"},4650:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>a,default:()=>u,frontMatter:()=>l,metadata:()=>o,toc:()=>d});var n=r(7462),i=(r(7294),r(3905));const l={id:"fields.fields_on_grid"},a=void 0,o={unversionedId:"API/fields.fields_on_grid",id:"API/fields.fields_on_grid",title:"fields.fields_on_grid",description:"fields.fieldsongrid",source:"@site/docs/API/fields.fields_on_grid.md",sourceDirName:"API",slug:"/API/fields.fields_on_grid",permalink:"/fmmax/API/fields.fields_on_grid",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fields.fields_on_grid.md",tags:[],version:"current",frontMatter:{id:"fields.fields_on_grid"},sidebar:"APISidebar",previous:{title:"fields.fields_on_coordinates",permalink:"/fmmax/API/fields.fields_on_coordinates"},next:{title:"fields.layer_fields_3d",permalink:"/fmmax/API/fields.layer_fields_3d"}},s={},d=[{value:"<code>fields.fields_on_grid</code>",id:"fieldsfields_on_grid",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],c={toc:d},f="wrapper";function u(e){let{components:t,...r}=e;return(0,i.kt)(f,(0,n.Z)({},c,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h3",{id:"fieldsfields_on_grid"},(0,i.kt)("inlineCode",{parentName:"h3"},"fields.fields_on_grid")),(0,i.kt)("p",null,"Transforms the fields from fourier representation to the grid."),(0,i.kt)("p",null,"The fields within an array of unit cells is returned, with the number of\ncells in each direction given by ",(0,i.kt)("inlineCode",{parentName:"p"},"num_unit_cells"),"."),(0,i.kt)("p",null,"The calculation is for a batch of fields, with the batch axis being the\nfinal axis. There can also be leading batch axes. Accordingly, fields\nshould have shape ",(0,i.kt)("inlineCode",{parentName:"p"},"(..., 2 * num_terms, num_amplitudes)"),". The trailing batch\ndimension is preferred because it allows matrix-matrix multiplication instead\nof batched matrix-vector multiplication."),(0,i.kt)("h4",{id:"args"},"Args:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"electric_field"),": ",(0,i.kt)("inlineCode",{parentName:"li"},"(ex, ey, ez)")," electric field Fourier amplitudes."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"magnetic_field"),": ",(0,i.kt)("inlineCode",{parentName:"li"},"(hx, hy, hz)")," magnetic field Fourier amplitudes."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"layer_solve_result"),": The results of the layer eigensolve."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"shape"),": The shape of the grid."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"num_unit_cells"),": The number of unit cells along each direction.")),(0,i.kt)("h4",{id:"returns"},"Returns:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"None"),": The electric field ",(0,i.kt)("inlineCode",{parentName:"li"},"(ex, ey, ez)"),", magnetic field ",(0,i.kt)("inlineCode",{parentName:"li"},"(hx, hy, hz)"),",\nand the grid coordinates ",(0,i.kt)("inlineCode",{parentName:"li"},"(x, y)"),".")))}u.isMDXComponent=!0}}]);