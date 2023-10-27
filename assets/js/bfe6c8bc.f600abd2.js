"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7094],{3905:(e,t,r)=>{r.d(t,{Zo:()=>d,kt:()=>f});var a=r(7294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function o(e,t){if(null==e)return{};var r,a,n=function(e,t){if(null==e)return{};var r,a,n={},i=Object.keys(e);for(a=0;a<i.length;a++)r=i[a],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)r=i[a],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var s=a.createContext({}),p=function(e){var t=a.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):l(l({},t),e)),r},d=function(e){var t=p(e.components);return a.createElement(s.Provider,{value:t},e.children)},u="mdxType",c={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,i=e.originalType,s=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),u=p(r),m=n,f=u["".concat(s,".").concat(m)]||u[m]||c[m]||i;return r?a.createElement(f,l(l({ref:t},d),{},{components:r})):a.createElement(f,l({ref:t},d))}));function f(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=r.length,l=new Array(i);l[0]=m;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[u]="string"==typeof e?e:n,l[1]=o;for(var p=2;p<i;p++)l[p]=r[p];return a.createElement.apply(null,l)}return a.createElement.apply(null,r)}m.displayName="MDXCreateElement"},8795:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>l,default:()=>c,frontMatter:()=>i,metadata:()=>o,toc:()=>p});var a=r(7462),n=(r(7294),r(3905));const i={id:"fields.amplitudes_interior"},l=void 0,o={unversionedId:"API/fields.amplitudes_interior",id:"API/fields.amplitudes_interior",title:"fields.amplitudes_interior",description:"fields.amplitudes_interior",source:"@site/docs/API/fields.amplitudes_interior.md",sourceDirName:"API",slug:"/API/fields.amplitudes_interior",permalink:"/fmmax/API/fields.amplitudes_interior",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fields.amplitudes_interior.md",tags:[],version:"current",frontMatter:{id:"fields.amplitudes_interior"},sidebar:"APISidebar",previous:{title:"fields.amplitude_poynting_flux",permalink:"/fmmax/API/fields.amplitude_poynting_flux"},next:{title:"fields.colocate_amplitudes",permalink:"/fmmax/API/fields.colocate_amplitudes"}},s={},p=[{value:"<code>fields.amplitudes_interior</code>",id:"fieldsamplitudes_interior",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],d={toc:p},u="wrapper";function c(e){let{components:t,...r}=e;return(0,n.kt)(u,(0,a.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h3",{id:"fieldsamplitudes_interior"},(0,n.kt)("inlineCode",{parentName:"h3"},"fields.amplitudes_interior")),(0,n.kt)("p",null,"Computes the wave amplitudes at an interior layer within a stack."),(0,n.kt)("p",null,"The calculation is for a batch of amplitudes, with the batch axis being the\nfinal axis. There can also be leading batch axes. Accordingly, amplitudes\nshould have shape ",(0,n.kt)("inlineCode",{parentName:"p"},"(..., 2 * num_terms, num_amplitudes)"),". The trailing batch\ndimension is preferred because it allows matrix-matrix multiplication instead\nof batched matrix-vector multiplication."),(0,n.kt)("h4",{id:"args"},"Args:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"s_matrix_before"),": The scattering matrix for the substack before the layer."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"s_matrix_after"),": The scattering matrix for the substack after the layer."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"forward_amplitude_0_start"),": The forward-propagating wave amplitude at the\nstart of the first layer of the stack."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"backward_amplitude_N_end"),": The backward-propagating wave amplitude at the\nend of the last layer of the stack.")),(0,n.kt)("h4",{id:"returns"},"Returns:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"None"),": The forward- and backward-propagating wave amplitude in the layer, defined\nat layer start and end, respectively.")))}c.isMDXComponent=!0}}]);