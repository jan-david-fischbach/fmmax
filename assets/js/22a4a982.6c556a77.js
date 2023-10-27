"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[834],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>f});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function c(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),l=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=l(e.components);return n.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,u=c(e,["components","mdxType","originalType","parentName"]),p=l(r),m=o,f=p["".concat(s,".").concat(m)]||p[m]||d[m]||a;return r?n.createElement(f,i(i({ref:t},u),{},{components:r})):n.createElement(f,i({ref:t},u))}));function f(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=m;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c[p]="string"==typeof e?e:o,i[1]=c;for(var l=2;l<a;l++)i[l]=r[l];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},4202:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>d,frontMatter:()=>a,metadata:()=>c,toc:()=>l});var n=r(7462),o=(r(7294),r(3905));const a={id:"sources.dirac_delta_source"},i=void 0,c={unversionedId:"API/sources.dirac_delta_source",id:"API/sources.dirac_delta_source",title:"sources.dirac_delta_source",description:"sources.diracdeltasource",source:"@site/docs/API/sources.dirac_delta_source.md",sourceDirName:"API",slug:"/API/sources.dirac_delta_source",permalink:"/fmmax/API/sources.dirac_delta_source",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/sources.dirac_delta_source.md",tags:[],version:"current",frontMatter:{id:"sources.dirac_delta_source"},sidebar:"APISidebar",previous:{title:"sources.amplitudes_for_source",permalink:"/fmmax/API/sources.amplitudes_for_source"},next:{title:"sources.emission_matrix",permalink:"/fmmax/API/sources.emission_matrix"}},s={},l=[{value:"<code>sources.dirac_delta_source</code>",id:"sourcesdirac_delta_source",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],u={toc:l},p="wrapper";function d(e){let{components:t,...r}=e;return(0,o.kt)(p,(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h3",{id:"sourcesdirac_delta_source"},(0,o.kt)("inlineCode",{parentName:"h3"},"sources.dirac_delta_source")),(0,o.kt)("p",null,"Returns the coefficients for a delta source at the specified location."),(0,o.kt)("p",null,"This function is appropriate for creating sources to be used with\n",(0,o.kt)("inlineCode",{parentName:"p"},"amplitudes_for_source"),"."),(0,o.kt)("h4",{id:"args"},"Args:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"location"),": The location of the source, with shape ",(0,o.kt)("inlineCode",{parentName:"li"},"(num_sources, 2)")," and\nthe trailing axis giving the x and y location. By convention, the\ncenter of the unit cell is at ",(0,o.kt)("inlineCode",{parentName:"li"},"(0, 0)"),"."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"in_plane_wavevector"),": The in-plane wavevevector for the calculation, which\ngives the offset of the plane wave decomposition. Has shape ",(0,o.kt)("inlineCode",{parentName:"li"},"(..., 2)"),"\nwith possible batch dimensions."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"primitive_lattice_vectors"),": The primitive lattice vectors of the unit cell."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"expansion"),": The Fourier expansion used for the calculation.")),(0,o.kt)("h4",{id:"returns"},"Returns:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"None"),": The coefficients, with the shape ",(0,o.kt)("inlineCode",{parentName:"li"},"(..., expansion.num_terms, num_sources)"),".")))}d.isMDXComponent=!0}}]);