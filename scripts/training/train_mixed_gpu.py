import os
import sys
import json
import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.dataset import TinyShakespeareDataset
from src.training.config import TrainingConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW

def evaluate_val_loss(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            if steps >= 20:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("     M0.1-Lite: Custom Text + Quijote Mix GPU Training (5000 Steps)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Prepare Text Corpora
    custom_story_path = "data/custom_story.txt"
    if not os.path.exists(custom_story_path):
        print("Warning: Custom story not found at data/custom_story.txt. It will be generated from request.")
        
    quijote_path = "data/quijote.txt"
    if not os.path.exists(quijote_path):
        print(f"Error: Quijote dataset not found at {quijote_path}")
        sys.exit(1)
        
    # Read Quijote and the Custom Story
    quijote_text = open(quijote_path, encoding="utf-8").read()
    split_idx = int(len(quijote_text) * 0.90)
    val_text = quijote_text[split_idx:]
    
    # We will get the custom story text directly from user request
    custom_story_text = """El sol se derramaba como un río de fuego frente al campo de batalla, tiñendo la hierba y reflejando la sangre derramada con un rojo profundo. Entre el estruendo de la guerra, el choque de espadas resonaba como un trueno. Un joven de ojos marrones, jadeante, se mantenía firme frente a su rival, un caballero experimentado que atacaba sin descanso, sus espadas se encontraban, acero contra acero, desgarrando el aire.

Los cuerpos chocaban, el metal cortaba sus carnes, y por un instante el mundo pareció detenerse. Una chispa de luz anaranjada iluminó los ojos del joven; algo cambió en ellos, un brillo imposible, como si la vida misma se escapase de dentro. Ambos se miraron fijamente, la sangre brotó de la boca del joven, un susurro salió de sus labios, inaudible para todos salvo para aquel frente a él.

La sangre se derramó sobre su pecho y sus movimientos se detuvieron, como si el tiempo vacilara. El caballero cerró los ojos y cayó, desvaneciéndose entre el atardecer. Y allí, en la lejanía, alguien lo miraba. No intervino, no respiró, pero registró cada detalle.

El silencio que siguió fue tan profundo como el último latido del joven. La imagen del campo bañado de rojo se desvaneció como humo y polvo hasta apagarse, seguido de un...
Negro absoluto.

Y entonces, un sonido leve.

Clink.

La porcelana tembló.

Drack abrió los ojos.

La taza de té se le resbaló de las manos y cayendo sobre el suelo de madera, derramándose en un charco dorado, un sobresalto que lo tenso por un instante, la sensación del acero entrando en su pecho todavía ardía ahí, donde no había herida… ni explicación, aquel no era el, entonces por que aquel vació no desaparece.

—Dios… ¿qué fue eso? —susurró, casi para sí mismo.

La puerta se abrió con prisa.

—¡Señor Drack! —María, la criada, corrió a verlo—. Señor, ¿está bien? Escuché un ruido fuerte.

—Estoy bien, María —respondió él sin mirarla, forzando una sonrisa—. Solo ha sido un descuido sin importancia.

Ella dudó, pero asintió.

Drack se levantó y se acercó a la ventana, fuera, un viejo manzano cargado de frutos se sacudía suavemente con la brisa, pájaros pequeños picoteaban entre las hojas, ajenos al mundo.

Intentó centrarse en la escena, en lo simple, en lo cotidiano. Pero la imagen del joven de ojos marrones —de sus ojos— seguía ardiendo en su mente, nítida, demasiado nítida para ser un sueño.

Ese susurro.

Esa luz imposible.

Drack apoyó una mano en el vidrio, como si buscara algo en el reflejo.
—¿Que fue lo que vi? —murmuró.
Solo el cantar tranquilo de los pájaros respondió…
Drack deslizo sus dedos por el vidrio, cayendo en un torbellino de pensamientos.
Un golpeteo suave interrumpió su hilo de pensamiento.
María regresó, pero no traía la bandeja ni la tetera.
—Señor Drack... —dijo con un tono distinto, más tenso que antes—. El señor August lo llama.
Drack parpadeó.
—¿Mi padre?
María asintió.
—Sí. Ordenó que se presentara decente y pidió que lo acompañara de inmediato.
Antes de que Drack pudiera responder, otra criada apareció por el pasillo, apresurada.
—María, el señor August quiere que Drack esté listo en diez minutos —informó, casi sin mirarla—. Y dijo que no tolerará retrasos hoy.
Drack soltó un suspiro. August no era un hombre brusco, pero sus llamados nunca eran casuales.
Menos aún si pedía formalidad.

María se inclinó suavemente y extendió la mano. —Venga, señor. Su padre lo espera, y… bueno, ya sabe cómo es él cuando prepara algo importante.
Drack asintió y la siguió por el corredor. Los pisos crujían bajo sus pasos, la casa entera olía a madera antigua, aceite de lámparas y un humo suave.
María abrió la puerta de su habitación. —Póngase cómodo, señor. Yo traeré lo necesario.
El cuarto de Drack era sencillo pero amplio: una cama de madera oscura, un escritorio lleno de papeles y bocetos, una ventana desde donde podía verse parte de la ciudad. A lo lejos, una columna de humo blanco se elevaba, señal de que un nuevo tren acababa de partir, lo que siempre le fascinaba.

María regresó con un traje cuidadosamente doblado. —Su padre pidió que usara traje —ella lo sostuvo con cuidado, extendiendo el chaleco gris oscuro y la camisa blanca recién planchada—. Dijo que hoy era… “un día para mostrarse como corresponde”.

Drack no preguntó qué quería decir con eso. No valía la pena. August siempre hablaba así cuando algo importante estaba por suceder, mientras María lo ayudaba a colocarse la camisa y ajustar los botones, Drack notó que sus manos seguían tensas. No por la visión, sino por una inquietud indescriptible. Que seria esto tan importante, ¿estaré ala altura de lo que busca padre?.

—¿Algún problema, señor Drack? —preguntó María al ver su expresión.
Drack negó suavemente. —No… solo estoy pensando demasiado.
—Bueno —sonrió ella, ajustando el cuello del chaleco—. Entonces permita que lo piense bien vestido.
Cuando terminó, dio un paso atrás. —Listo, señor. Impecable. Su padre lo espera en el salón principal.
Drack respiró hondo y se miró un momento en el espejo, vio a un joven que intentaba parecer calmado. —Bien —dijo finalmente—. Vamos.
Salió a enfrentar lo que su padre tenía preparado. El camino hacia el salón principal no era largo, pero Drack lo sintió eterno. A medida que avanzaba, notó que no lo llevaban al salón común donde August solía reunirse con comerciantes. No. María tomó un pasillo distinto, uno que casi nunca se usaba, con alfombras más antiguas y lámparas de gas que iluminaban con un tono cálido.

Drack frunció el ceño. Ese pasillo conducía a un solo lugar. El Salón del Este. Un cuarto reservado para ocasiones demasiado formales… o demasiado graves.
María se detuvo frente a las puertas dobles de madera oscura. —Están ahí dentro, señor —dijo en voz baja—. Su padre y… un invitado.
Drack sintió un nudo en la garganta. —¿Un invitado?
—Sí. Llegó hace unos minutos. Es alguien muy importante… por lo que he alcanzado a oír.
Antes de que pudiera preguntar más, María abrió una rendija de la puerta y la empujó suavemente. Un murmullo de conversación escapó del interior, junto a un aroma a té fuerte y algo más… un olor que se impregnaba en el ambiente.
Drack entró.

El salón estaba en penumbra, iluminado solo por un ventanal alto donde la luz caía sobre alfombras bordadas y muebles de madera. Era un cuarto construido para impresionar. Y funcionaba.
En el centro, dos sillones largos se enfrentaban, separados por una mesa de cristal. August estaba sentado en uno, erguido, con su porte severo habitual y el bigote perfectamente recortado. Sin embargo, no parecía relajado; sus dedos golpeteaban el reposabrazos, un tic que solo aparecía cuando algo lo inquietaba.
Frente a él, con la calma de un hombre acostumbrado a ser el centro de atención, estaba el invitado. Un sombrero de copa alto descansaba sobre su cabeza y portaba un bastón cuyo mango de plata tenía la forma de un ojo extraño. Su traje era impecable.
El hombre levantó la vista al verlo. Tenía ojos afilados… demasiado atentos. Parecían registrar cada detalle, como si estuviera tomando notas invisibles.
Drack se tensó sin saber por qué. Ese no era un visitante cualquiera.
August lo llamó con una seña de la mano. —Drack. Llegas justo a tiempo.
Ambos esbozaron sonrisas nerviosas. El invitado dejó su taza en el plato con un toque suave y se levantó, ofreciendo una leve inclinación.

—Así que este es su hijo —su voz era profunda, cálida, pero con un filo difícil de describir—. Mucho gusto, joven Drack. He oído… cosas interesantes sobre usted.
Drack tragó saliva. No supo qué responder y se ajustó el cuello del chaleco, un acto reflejo para sentirse “correcto” frente a su padre. Pero aquel hombre lo estaba evaluando. Como si midiera algo que el joven no sabía que poseía.
August intervino. —Drack, permíteme presentarte al señor Sil Wornhilt. Empresario, propietario de la mayor fábrica de vías ferroviarias en la región. Y… —hizo una pausa respetuosa— subcapitán en el Cuerpo de Investigación de la ciudad.
Sil sonrió apenas. Lo justo, lo limpio. Nada más. —No se preocupe, no he venido por motivos policiales —aclaró con tono ligero—. Solo por… oportunidades.
Un leve escalofrío recorrió la espalda de Drack. No era miedo, sino la certeza de que ese hombre sabía más de lo que mostraba. Más de lo que cualquiera debería.
August, intentando recuperar el control de la conversación, habló con orgullo. —Mi hijo es aplicado, inteligente. Aprende rápido, demasiado rápido dicen algunos, y sabe observar —agregó con una sonrisa nerviosa—. Tiene… talento natural.
Sil giró apenas la cabeza. No hacia August, sino hacia Drack. Lo observó como quien lee un texto oculto entre líneas y lo interrumpió sin pedir permiso: —Buscamos nuevos talentos para la división de investigaciones —dijo sin rodeos, con voz precisa como un corte limpio—. Jóvenes capaces de… ver más allá de lo evidente.
August abrió la boca para decir algo, pero Sil no le dio espacio. Inclinó la cabeza hacia el joven, un movimiento premeditado, como si ajustara la distancia exacta para decir algo confidencial.
—Joven Drack… —su voz bajó, fina como un hilo tesso—. Usted tiene un olor inconfundible a muerte.
Hubo una pausa que no pertenecía a la habitación, ni al momento.
—Resonante.
Drack sintió cómo el aire se le quedaba atrapado en la garganta. No entendió la palabra. No entendió nada. Pero el mundo pareció apretarse a su alrededor, como si la luz del salón hubiera perdido temperatura.
Sil se enderezó con la misma calma con la que alguien acomoda un libro. —Un talento raro —añadió en voz normal, tomando su taza de té—. Muy útil, si se cultiva.
El joven no pudo responder. Ni moverse. August tampoco dijo nada, solo tensó la mandíbula. Por primera vez desde que entró, Drack sintió que estaba frente a alguien que lo había visto antes de que él mismo supiera quién era.
Sil dejó la taza con un clink delicado, casi elegante. —Señor August —dijo con tono neutral—, retomemos lo que importa.
El cambio fue tan brusco que Drack parpadeó. Su padre se movió incómodo, carraspeó fuerte y levantó una mano. —Drack… —forzó una sonrisa cargada de tensión— ven, siéntate aquí, a mi lado.
Drack obedeció. Sil siguió cada movimiento, no con intriga, sino con precisión calculada.
August cruzó una pierna sobre la otra. —Bien, señor Sil… ¿qué asuntos lo traen exactamente a mi casa?
Sil sonrió apenas. No era cortesía. Era satisfacción. Metió una mano en el bolsillo interior de su chaqueta y sacó un pequeño reloj de cadena. Antes de abrirlo, deslizó entre sus dedos una ficha circular, metálica, pulida como si jamás hubiese tocado una mano humana. Dejando la caer sobre la mesa.
Tac.
El sonido fue seco, pesado. Más pesado de lo que una simple moneda podría hacer. Drack la observó, su brillo no era el del cobre, ni plata, ni oro. Y los grabados no eran símbolos que él conociera. Sil apoyó un dedo sobre ella, girándola con suavemente.
—No es una moneda —dijo—. Es una ficha bancaria. Una llave.
August frunció el ceño. —¿Llave… de qué?
Sil levantó la vista. Sus ojos, antes atentos, ahora parecían vacíos. —De mi bóveda privada —pausó un segundo—. Y de una propuesta lo suficientemente grande como para que yo cruzara la ciudad personalmente.
August contuvo el aliento. Drack sintió un tirón en el estómago, como si un hilo invisible le presionara.
—Pero antes de explicarlo —continuó Sil, retirando el dedo de la mesa—, debo saber algo, señor August. Algo simple: ¿Está dispuesto a considerar una asociación… incluso si requiere discreción absoluta?
Los ojos de Sil se desviaron hacia Drack. No para observarlo, sino para recordarle que seguía ahí. Que todo, de alguna manera, lo involucraba. August intentó responder, pero Sil levantó una mano con un gesto suave que encerraba autoridad.
—Permítame mostrarle primero lo que ofrezco.
Tomó la ficha y la colocó frente a August.
La luz del ventanal se reflejó en sus bordes, revelando grabados finos como venas metálicas. —Con esta ficha, usted tendrá acceso directo a una línea de crédito exclusiva, respaldada por mis propias reservas. Capital líquido. Inmediato. Sin intermediarios. Sin límites impuestos por el banco central.
August abrió ligeramente los ojos. Eso no era una oferta. Era una puerta directa a un poder financiero al que ningún empresario de rango medio podía aspirar.
Sil sonrió al ver su reacción. —Con mi respaldo, su empresa de exportación podría duplicar o triplicar su alcance en menos de un año. Nuevas vías, nuevos trenes, nuevas rutas. Incluso acceso a maquinaria con la que no ha soñado. Todo eso, señor August, está a unas pocas palabras de distancia.
El padre de Drack tragó saliva de forma audible. La habitación parecía encogerse. Sil cruzó una pierna con calma, como quien aún no había dicho lo realmente importante. —Pero, como comprenderá, un apoyo como este requiere algo equivalente a cambio. Algo que no puede ofrecerme el dinero.
Los ojos de Sil se movieron lentamente hacia el joven, con la certeza tranquila de quien ya sabe la respuesta. —Quiero a su hijo.
August se tensó de inmediato. Sil continuó antes de que cualquiera pudiera protestar. —No como esclavo, no como sirviente. Como funcionario oficial del Cuerpo de Investigación. Bajo mi supervisión directa.
August apretó los puños. —Señor Sil… mi hijo no tiene formación para eso.
—La tendrá —respondió el invitado, imperturbable—. Lo entrenaremos. Lo prepararemos. Recibirá pago completo, honorarios, alojamiento temporal si es necesario. Todo lo que un miembro oficial del cuerpo debe recibir. —Inclinó la cabeza—. Con una sola condición adicional.
El silencio cayó sobre la sala. Sil dejó caer el peso de cada palabra: —El joven Drack no podrá renunciar hasta que yo lo autorice. Ni retirarse. Ni desaparecer. Ni intentar abandonar el cuerpo sin mi consentimiento.
Sil no sonreía ahora. Su mirada era hielo puro. —Una inversión como esta requiere seguridad. Y él representa un… valor que prefiero asegurar personalmente.
August sostuvo la ficha entre los dedos como quien carga una piedra demasiado pesada. La mano le temblaba apenas. Drack sintió que la garganta le ardía. No sabía si estaba siendo comprado, contratado… o entregado.
Sil se recostó en el sillón con la calma de un depredador saciado. —Piénselo con cuidado. Lo que ambos ganan es enorme. Lo que pierden… —dejó la frase suspendida— depende de sus decisiones.
Un sudor frío recorrió la espalda de Drack. Miró a su padre. No buscaba permiso, ni siquiera valentía. Buscaba una señal de que August sabía qué hacer. Un refugio.
Pero August no lo miró. Seguía observando la ficha como si contuviera un mañana que no podía rechazar… y un precio que no podía asumir.
Por primera vez en años, Drack descubrió que su padre no tenía la respuesta. Y ese descubrimiento lo dejó expuesto.

Hace 5 días. Oficina del Subcapitán Wornhilt.
Una tormenta azotaba la ciudad, convirtiendo las calles en ríos de lodo.
Dentro, el sonido del aguacero golpeando el cristal era lo único que llenaba el silencio.
Sil Wornhilt ajustó la llama de la lámpara de aceite sobre su escritorio, iluminando una torre de carpetas. Treinta años de servicio le habían enseñado que la mayoría de los "talentos excepcionales" eran decepcionantes, trucos sin importancia, solo para llamar la atención. Nobles sin escrúpulos, aquellos que conocían el archivo "Talentos excepcionales".
Sil pasó otra hoja con desgano. Y entonces, se detuvo.
Entonces, vio un nombre.
Sil se detuvo en seco.
No podía ser.
Se levantó de golpe, derribando la silla. Ignoró el orden, ignoró la calma. Buscó frenéticamente en su estante privado, sus dedos temblando ligeramente mientras lanzaba papeles al suelo hasta dar con lo que buscaba.
Dos carpetas viejas, polvorientas, olvidadas por todos menos por él.
Ambas llevaban el mismo encabezado: Drack Vans.
Un joven noble de linaje desgastado, irrelevante para la sociedad actual. Pero los reportes decían otra cosa. Sil abrió el primero, fechado hace trece años. La tinta estaba algo corrida, pero las palabras del investigador de campo eran legibles y cortantes.
Reporte N.º 402 - Edad del sujeto: 4 años. Observación de campo: Casa de los nobles Vans. "El sujeto muestra una comprensión anómala del ciclo vital. Se le observó dando muerte a pequeños animales (aves de corral, roedores), pero también curando a otros con heridas graves. Lo inquietante no es el acto, sino la reacción de la fauna. Los animales no huyen de él. Se acercan. Aceptan su toque, sea para sanar o para morir, con una docilidad antinatural. No hay miedo. Solo silencio."
Sil tragó saliva y abrió la segunda carpeta con brusquedad.
Reporte N.º 789 - Edad del sujeto: 9 años. Observación de campo: Academia de Esgrima Local. "El sujeto supera el apartado cognitivo promedio, pero su desempeño físico es inexplicable. Durante la práctica, desarmó a un joven instructor. No usó las formas enseñadas en la academia. Utilizó un estilo de guardia baja y estocadas al cuello que no se ve en los manuales modernos. Esgrima antigua. Movimientos de un soldado de hace siglos. Al ser interrogado, el niño no recordaba haberlo aprendido."
Sil cerró las carpetas de golpe, respirando con dificultad. El sonido de la lluvia parecía más fuerte ahora.
—Esgrima antigua... animales que aceptan la muerte... —susurró Sil, pasando una mano por su rostro, sintiendo cómo el fantasma de su alumna volvía a acecharlo desde las sombras de su memoria.
But faltaba una pieza. El motivo de su frenesí actual. El tercer reporte. El que acababa de llegar esa misma mañana y que descansaba sobre su escritorio, aún húmedo por la lluvia del mensajero.
Drack Vans, 17 años.
Sil dejo caer los archivos, levantando una nube de polvo que danzó bajo la luz de la lámpara. Sus manos, usualmente firmes como el acero, temblaban apenas al alcanzar el tercer documento. El papel aún estaba húmedo por la lluvia, traído por un mensajero hace de unas horas.
Suspiro, su mano se deslizo por su rostro. Lentamente deslizo un cajón de su escritorio, Tomo sus gafas, ajustando los documentos.
Reporte de Incidente N.º 1444 - Urgente. Agente de Campo: J. L. (Turno nocturno). Hora: 9:45 PM.
"Regresaba de una vigilancia rutinaria sin resultados. Al pasar por el perímetro de la casa Vans, escuché un sonido inusual. No era viento. Era madera. Un crujido largo, como si algo se estuviera rompiendo o estirando.
salte el muro para investigar. Lo que vi no tiene sentido.
El hijo de los Vans. Drack Vans, estaba en el jardín trasero. Vestía ropa de dormir. Parecía sonámbulo o en un estado de trance profundo. Estaba abrazado al viejo manzano seco que la familia nunca taló. Mientras lo sostenía, el árbol no se rompía. Cambiaba. La corteza gris recobraba color. Las ramas secas se estiraban con chasquidos audibles y brotes verdes surgían.
No hubo destellos cegadores ni ruidos explosivos. Solo un silencio pesado y la vida regresando a donde esta se marchitaba. El sujeto soltó el árbol y volvió a entrar a la casa, tambaleándose, aparentemente sin percatarse del evento."
Sil dejó el papel sobre la mesa. El sonido de su propia respiración le pareció ensordecedor.
Se quitó las gafas y se frotó los ojos cansados.
—No veo un talento así desde aquella vez... —murmuró, y la imagen de ella, su antigua alumna, cruzó su mente como un relámpago doloroso—.
—¿Por qué? —murmuró, frotándose la cara—. ¿Por qué ahora?
Pero no había tiempo para lamentos. Si un guardia raso lo había visto, el rumor ya estaba corriendo. Y en esta ciudad, los rumores viajan rápido. Si la Iglesia se enteraba de que un crío de diecisiete años podía revertir la muerte... Drack Vans no llegaría vivo al fin de mes.
Sil se levantó de golpe. El frenesí había desaparecido, reemplazado por una determinación fría y absoluta.
Tomó su bastón con empuñadura de ojo de plata. Agarró su sombrero de copa.
Se puso de pie, ya no con frenesí, sino con la urgencia fría de quien tiene una misión. Caminó hacia la puerta. —Preparen un contrato financiero —ordenó a su asistente—. Y redacten una transferencia para el Cuerpo de Investigación. Iré a la casa de los Vans.
Pero no podía ir con las manos vacías. Necesitaba un trato irrechazable. — Y traigan la ficha bancaria de acceso ilimitado — .
Iba a comprar a ese chico. Iba a protegerlo. O iba a condenarlo. Pero nadie más lo tendría.
Sil salió de la oficina, dejando atrás los reportes y el eco de la lluvia, directo hacia los preparativos, Nadie mas podría tener al chico Vans.

Presente. Salón del Este.
El silencio en la habitación se había vuelto insoportable. August Vans sostenía la ficha metálica, pero sus ojos iban de la pieza de metal a su hijo, y luego... al suelo. Sus manos temblaban. La oferta era poder puro, pero el costo... el costo era su sangre.
August abrió la boca. Drack vio en los ojos de su padre la negativa formándose. Iba a rechazarlo. El miedo a perder a su hijo, o quizás un remanente de orgullo, estaba ganando la batalla.
—Señor Sil... —empezó August, con la voz quebrada, devolviendo la ficha lentamente hacia la mesa—. Es... es demasiado. No puedo...
Sil no se inmutó. Lo había previsto. Antes de que la ficha tocara la madera, Sil habló. Su voz cortó el aire como una suave brisa.
—Entiendo su duda, August. —Sil se inclinó hacia adelante, bajando el tono a uno de confidencia —. Hablemos de números reales. Usted espera que, a cambio de este respaldo y de la maquinaria, yo exija el control mayoritario de su empresa. Or al menos la mitad.
August se detuvo. Era exactamente lo que pensaba. Un 50% era el minimo para un tratado de ese calibre. Era entregar la empresa.
Sil sonrió, una curva leve y astuta en sus labios. —Mi interés no es su empresa, August. Mi interés es el talento que se desperdicia en esta sala. Por eso, mi condición final es simple: A cambio del capital y el crédito ilimitado... el Cuerpo de Investigación solo tomará el 15% de las ganancias netas.
August se quedó en blanco. Parpadeó, aturdido, como si le hubieran golpeado en la cara. —¿Quince...? —repitió en un susurro incrédulo.
—Quince —confirmó Sil—. El 85% restante es suyo. Para su legado. Para su familia.
El aire cambió en la habitación. Drack vio el momento exacto en que su padre se quebró. Ya no había duda, ni orgullo. Solo un hambre voraz. No era una oferta, era un regalo caído del cielo. O eso veía en los ojos de su padre.
La mano de August se cerró con fuerza alrededor de la ficha. El temblor desapareció. —Trato hecho —susurró, casi sin aliento.
Sil asintió, satisfecho. Se puso de pie con elegancia, alisándose el traje. —Sabía que entraría en razón.
Sacó un sobre grueso de su chaqueta y lo dejó sobre la mesa, junto a la ficha. —Aquí está el contrato preliminar. Léalo. Y fírmelo.
Sil caminó hacia Drack y, por un instante, puso una mano sobre su hombro. El toque fue pesado, firme. —No se preocupe por el equipaje ahora, joven Drack. Disfrute sus últimos días de... normalidad.
Se volvió hacia August, quien seguía mirando la ficha como si fuera un ídolo sagrado. —Enviaré a alguien por el muchacho y por el contrato firmado dentro de tres días. Tengan todo listo.
Sin esperar despedidas, Sil giró sobre sus talones, hizo sonar su bastón contra el suelo y salió por las puertas dobles, dejando atrás un silencio que pesaba más que antes.
Drack miró a su padre. August no levantó la vista. Solo apretaba la ficha en su palma, con los ojos húmedos y brillantes de codicia y alivio.
El sonido de la puerta cerrándose tras Sil fue el detonante.
Drack sintió que algo se rompía dentro de él, no fue un hueso, ni algo físico. Fue el respeto. Esa veneración silenciosa y temerosa que había mantenido por August durante diecisiete años se evaporó en un instante, dejando solo una ceniza amarga.
Llevó la mano a su cuello. Ese cuello que minutos antes había ajustado con tanto esmero para ser el "hijo perfecto".
—¿Eso es todo? —preguntó Drack. Su voz no tembló. Salió rasposa, como nunca antes.
August seguía hipnotizado por el metal en su mano. —Es el quince por ciento, Drack. ¿Tienes idea de lo que eso significa? Es el futuro. Es...
—¡Es mi vida! —El grito desgarró el silencio del salón, rebotando en las paredes altas.
Fue la primera vez que Drack Vans alzaba la voz.
August lo miró, atónito, con la boca entreabierta. Pero Drack ya no estaba allí para escuchar sermones. Se arrancó la corbata con un tirón violento, escuchando cómo la tela fina se rasgaba levemente, y la arrojó con desprecio. La prenda de seda, símbolo de su obediencia, cayó flácida sobre el regazo de su padre, cubriendo la preciada ficha.
—Me vendiste —escupió Drack, con el pecho agitado—. Por rutas de trenes y maquinaria. Por dinero.
—¡Lo hice por esta familia! —bramó August, recuperando su autoridad y poniéndose de pie, rojo de ira—. ¡Sin esto estaríamos arruinados en pocos años! ¡Deberías agradecerme! ¡Te conseguí un puesto que otros matarían por tener!
Drack dio un paso atrás. La decepción en era más dolorosa que cualquier golpe. —No —dijo en voz baja, fría—. Lo hiciste por ti. Por tu codicia.
Drack se dio la vuelta y comenzó a caminar hacia la salida. —A partir de ahora, August... no te considero mi padre.
August golpeó la mesa con el puño. —¡¿Cómo te atreves?!
Drack se detuvo bajo el umbral de las puertas dobles, pero no se giró. —Disfruta tu dinero. Pero si alguna vez me necesitas... si alguna vez te enfermas o te arrepientes... no me busques. No me dirijas la palabra. Para ti, yo morí en esta habitación.
August, temblando de furia, agarró la corbata de Drack y la lanzó al suelo. —¡Bien! —gritó, su voz quebrándose por la rabia—. ¡Entonces vete! ¡Ya no perteneces a los Vans! ¡Me alegra que te largues en tres días! ¡Lárgate y no vuelvas!
Drack no respondió. Simplemente cruzó el umbral.
Al salir al pasillo, se encontró con María. Estaba pálida, con las manos cubriendo su boca, sus ojos llenos de lágrimas. Ella intentó acercarse, consolarlo, pero Drack camino de frente. No podía detenerse. Si se detenía, se derrumbaría.
Caminó por el pasillo oscuro hacia su habitación. Las lágrimas comenzaron a desbordarse, calientes y silenciosas, nublando su vista.
Y entonces, lo sintió.
Un ardor repentino que le subía por el cuello. No era magia, ni un poder extraño. Era rabia pura. Sangre caliente agolpándose en su pecho y martilleando en sus sienes, una mezcla de furia y duelo que amenazaba con explotar.
Drack entró a su cuarto y cerró la puerta, dejándose caer contra la madera fría.
Se llevó la mano al pecho, apretando la tela de la camisa con fuerza, justo donde la piel escondía esa vieja cicatriz pálida. Sus dedos se clavaron allí, buscando contener el dolor. Al mismo tiempo, su hombro izquierdo le pesó, una molestia fantasma justo sobre aquel lunar oscuro, esa mancha de nacimiento que cargaba en la espalda como una sombra.
Nada brilló. Nada se movió. Las marcas siguieron allí, quietas, silenciosas, simples trazos en su piel... esperando.
—Maldita sea... —sollozó, deslizándose hasta el suelo.
En la soledad de su habitación, sintiéndose apagado como una estrella moribunda, Drack Vans lloró la muerte de la única vida que había conocido. Lloró por la familia que acababa de perder.

Los tres días siguientes fueron un desastre silencioso.
La habitación se había convertido en una celda de madera y humedad. El aire estaba estancado, impregnado de aquel frío cortante, colandose por esta madrugada. Drack nisiquiera había abierto las cortinas.
Se encontraba sentado en el borde de la cama, aquel era apenas una sombra de lo que fue. Tenía los pómulos levemente hundidos y unas ojeras oscuras le enmarcaban la mirada cansada. María le llevaba bandejas a diario, rogándole en susurros que comiera, pero él apenas probaba la mitad, tragando sin sentir el sabor, con la garganta seca.
Tock, tock.
—¿Joven Drack? ¿Se encuentra despierto? —La voz de María sonó amortiguada tras el roble.
Drack se levantó despacio, sintiendo un leve temblor en las piernas, y abrió sin decir palabra. María, que aguardaba con una bandeja en alto, casi pierde el equilibrio ante la brusquedad del movimiento, derramando un poco de té sobre el paño limpio.
—¡Ay! Jeje, lo siento... —María agachó la cabeza, entrando con pasos rápidos para dejar la bandeja sobre la mesa—. El té está cargado, señorito. Bébale, que le hace falta color en la cara.
Drack tomó la taza con manos entumecidas. Bebió un sorbo. El líquido bajó caliente y amargo, rascándole la garganta. Mientras él bebía mecánicamente, María comenzó a moverse por el cuarto, doblando camisas de lino y guardando cuadernos de bocetos en una maleta de cuero.
—¿Por qué marcharse así, señorito? —preguntó ella en un susurro tosco—. El señor August está... inmerso en esa oficina. Solo mira aquella ficha metálica. No levanta la vista ni para comer.
Drack bajó la taza. El sonido de la porcelana contra el plato fue seco.
—Oro, María. Eso significó para mi padre. Solo un par de monedas. Me cambió por capital para sus trenes, como quien vende un caballo viejo.
María se detuvo. Terminó de acomodar la ropa y, tras dudar un segundo, sacó un objeto envuelto en tela basta del bolsillo de su delantal.
—Tome. Era de mi hermano, que en paz descanse. Sé que ese lugar es peligroso, aunque digan que es de "investigación". En los caminos de la vida, nadie regala nada.
Drack desenvolvió el trapo. Era una daga de acero opaco, pesada. En el mango, los trazos toscos formaban un hombre sin ojos: el Dios Ciego. Drack la observó con profunda extrañeza. Sabía muy poco de aquel hermano muerto, de hecho, apenas sabía de la vida de María fuera de las paredes de esta casa. Aun así, apretó el mango frío, sintiendo el peso real del metal.
—Gracias, María.
Dos horas después, cargaba su propia maleta bajando las escaleras principales. Al pasar por el pasillo del ala este, sus pasos se ralentizaron. La puerta del despacho de August estaba cerrada a cal y canto. No hubo despedida.
No hubo un crujido de madera ni una voz llamándolo. El silencio de su padre fue la última herida. Drack apretó la mandíbula, giró el rostro y salió por la puerta principal, recibiendo el golpe helado del viento en la cara.
En la acera, el carruaje negro de Sil Wornhilt aguardaba. Apoyado contra la madera lacada, un hombre de unos veinticinco años lo esperaba. Vestía el uniforme gris acero del Cuerpo. Una cicatriz delgada le cruzaba desde el labio hasta la mejilla izquierda.
—¿Drack Vans? —preguntó el oficial. Su voz era seca.
Drack apretó el asa de su maleta.
—Solo Drack.
El hombre no dijo nada de inmediato. Sus ojos recorrieron al muchacho de arriba abajo. De repente, una mueca burlona deformó la cicatriz de su rostro y le abrió la puerta del carruaje con una inclinación exagerada y divertida.
—Un gusto, "Solo Drack". Soy Kael. Sube, el subcapitán no es un hombre paciente.
Drack subió sin mirar atrás. El carruaje arrancó de inmediato. El viaje fue un descenso gradual. La elegancia de los barrios nobles quedó atrás, reemplazada por un cielo que ya no era gris natural, sino opaco, asfixiado por el humo de las chimeneas industriales. Drack miraba por la ventana sin ver realmente. El silencio dentro de la cabina era interrumpido solo por el trote de los caballos y el traqueteo de las ruedas contra los adoquines irregulares.
—Escúchame —dijo Kael, borrando cualquier rastro de burla—. Sil cree que eres "especial". Tu entrenamiento empezará pronto y no será agradable. ¿Conoces la palabra "Resonante"? ¿A qué dioses conoces tú?
Drack parpadeó, sacándose a sí mismo del letargo.
—Conozco los nombres de las fábulas. El Ciego, la Dama del Hambre, el Viejo Rey... pero son solo cuentos. No creo en ninguno.
Kael lo observó en silencio y soltó una carcajada seca. Intentó lanzar una broma rápida sobre engranajes que no hablaban para aliviar la atmósfera, pero la respuesta de Drack fue un muro de indiferencia. Se detuvieron frente a una estructura cuadrada de dos plantas, en medio del ruido del distrito industrial.
—¿Esto es todo? —preguntó Drack al bajar—. Parece poco más que una oficina de atención.
—Las apariencias son el primer escudo —replicó Kael—. Sígueme.
Dentro, la recepción olía a cera y papel. Kael bromeó con la secretaria, quien presionó un mecanismo oculto. Una sección entera de la pared se deslizó hacia un lado revelando una escalera de piedra. Comenzaron a bajar. Piso -1. Piso -2. A medida que descendían, el aire perdía oxígeno y se volvía gélido.
Finalmente, llegaron al Piso -3. El aire aquí abajo apestaba a gas quemado de las lámparas que iluminaban las paredes de ladrillo. Pero entre el olor, Drack captó un rastro sutil, casi imperceptible, de perfume caro. Allí aguardaban.
Bajo la luz parpadeante, la primera figura que Drack reconoció fue a Sil Wornhilt. Recto, con su traje impecable y ambas manos apoyadas en el bastón del ojo de plata. A su lado, dándole la espalda a Drack, había un hombre de piel oscura y cabello blanco cortado al ras. Llevaba un bastón de madera negra. No se giró al escuchar los pasos.
—Soy tu Capitán —sentenció el anciano, con una voz que resonó en la piedra—. Aquí no hay nombres de cuna. Solo órdenes.
El hombre pasó de largo hacia las sombras del pasillo sin dedicarle a Drack una sola mirada. Drack se quedó paralizado.
—¿Qué diablos fue eso? —preguntó Drack, su voz tensa—. Ni siquiera me miró. Y... ¿no tiene nombre?
—Es el Capitán Abel —respondió Kael, sin detenerse—. Y créeme, chico, prefieres que no te mire. Si Abel te está prestando atención, significa que estás a punto de morir. Vamos arriba.
Regresaron al Piso -2. Kael empujó una pesada puerta de hierro. El aire apestaba a pólvora seca, aceite de armas y un toque rancio a tabaco y especias muertas. Era un taller caótico. En el centro, un anciano con gafas de relojero trabajaba sobre una mesa manchada de químicos.
—Zev —lo llamó Kael—. Tenemos sangre nueva. El subcapitán exige el pacto.
Zev arrastró los pies hasta un rincón y empujó un cuenco de hierro fundido sobre una mesa. El recipiente estaba lleno de un agua oscura y turbia. Flotando en la superficie había pólvora, ajo machacado, especias de color óxido y una pata de gallo picada.
—Alquimia básica de contención. Una promesa por Resonancia —explicó Kael—. Mete la mano.
Drack apretó la mandíbula y sumergió la mano derecha en el líquido frío. Kael hizo lo mismo. Bajo el agua, agarró la mano de Drack en un apretón firme. Zev murmuró algo y arrojó un polvo gris. Al instante, un hilo oscuro se formó en el agua, enroscándose alrededor de ambas manos. El hilo quemaba.
—La promesa es simple —dijo Kael—. No revelarás la existencia de la Resonancia de ninguno de los miembros de este cuerpo. Ni información alguna sobre nuestras operaciones. Si rompes la promesa, tu propia mano te será arrebatada desde el codo. ¿Lo juras?
—Lo juro.
El hilo se disolvió. Drack retiró la mano, sintiendo que la vieja cicatriz en su pecho daba un latigazo de dolor repentino.
—Listo. Ya eres propiedad del Cuerpo —Zev tosió, dándole la espalda—. O de El Manto, si prefieres el título de las sombras. Soy Zev. Resonante Rango IX. Camino de la Alquimia. Alquimista de la pólvora.
Kael se apoyó contra una caja de municiones y explicó:
—Para los civiles somos el Cuerpo de Investigación. Pero en las sombras, somos El Manto. Investigaron y contuvieron lo que el pueblo no debe ver.
Zev encajó un engranaje en su rifle con un golpe seco y, sin levantar la vista, escupió el resto:
—A partir del Rango X eliges un camino. Y aquí contamos hacia atrás. Mientras más bajo es tu número, más te acercas a los Apóstoles y a los mismísimos Dioses... y menos de ti mismo.
Kael se inclinó hacia Drack, analizándolo.
—Apenas siento tu energía o tu olor de Resonante. Aunque... apestas a muerte de una forma muy curiosa. ¿Acaso eres un pequeño bastardo psicópata en secreto?
Drack levantó una ceja, mirándolo con una extrañeza genuina, casi aburrida. —Solo estoy cansado —. Respondió con voz apagada—. Búscate a otro para tus chistes.
Kael soltó una carcajada seca.
—Okey, okey... Solo Dack.
La puerta de hierro chirrió. Sil Wornhilt entró y dejó caer un conjunto de ropa gris acero sobre una mesa.
—Cámbiate. Ese traje de noble ya no te sirve aquí. El Capitán Abel y yo iremos al centro de la ciudad. Solo nosotros dos. Tenemos un asunto que purgar.
Sil se detuvo un segundo en el umbral.
—Kael, enséñale el Piso -2. Es su nuevo hogar ahora. Mantenlo a raya. Volveremos al anochecer.
Sil cerró la puerta de hierro tras de sí con un golpe seco, dejando a Drack a solas con Kael y el viejo Zev.

Kael se quedó mirando la puerta cerrada durante un segundo, soltó un suspiro pesado y luego se giró hacia Drack con esa mueca que intentaba ser una sonrisa, pero que moría mucho antes de llegar a sus ojos cansados. El eco del portazo de Sil todavía vibraba en las paredes del taller, dejando un silencio denso, cargado del olor a queroseno y metal viejo.
—Bien, Drack. Bienvenido al sótano. Muévete, el viejo no tiene mucha paciencia hoy y el aire aquí abajo se vuelve rancio si te quedas quieto demasiado tiempo.
Caminaron por el pasillo del Piso -2. Era un corredor largo, de paredes de piedra húmeda y ladrillo visto, flanqueado por una docena de puertas de madera reforzada con herrajes de hierro que, a pesar de su solidez, tenían un acabado que recordaba a celdas de lujo. El aire aquí abajo era notablemente más pesado, una masa estancada que parecía presionar los pulmones con cada inhalación. Kael se detuvo casi al final del pasillo, señalando dos puertas contiguas.
—Estas son las tuyas —dijo Kael, entregándole un juego de llaves—. Tu dormitorio y la sala de estudio. Entremos a la de estudio; hay más espacio para que te desmayes cuando escuches lo que viene.
La habitación era austera: una mesa de roble macizo con marcas de quemaduras, un par de sillas y estanterías vacías. Drack dejó su maleta en el suelo, pero algo no le cuadraba en la arquitectura. Miró hacia el inicio del pasillo, donde la entrada principal de la planta quedaba sospechosamente cerca de su puerta.
—¿Por qué aquí? —preguntó Drack, frunciendo el ceño—. Hay muchas habitaciones vacías más allá. ¿Por qué ponerme tan cerca de la entrada si apenas vive gente aquí abajo?
Zev Grimm, que venía arrastrando los pies detrás de ellos, soltó un bufido mientras se apoyaba en el marco de la puerta.
—Porque en este pasillo solo respiran tres personas de forma permanente, Joven Vans —gruñó Zev—. Tú, la mujer esa, Valeria, y yo. Los demás tienen vidas de fachada allá arriba. Te ponemos cerca del acceso para que seas el primero en despertar si algo decide bajar por las escaleras sin invitación... o para que seas el primero al que le corten el cuello si Sil se aburre de ti.
Zev entró a la sala y clavó sus ojos tras el cristal grueso de sus gafas en el cinturón de Drack. Olfateó el aire como un sabueso.
—Saca esa chatarra que llevas escondida —ordenó el viejo.
Drack dudó, pero sacó la daga de María. La hoja de acero opaco con la efigie del Dios Ciego pareció absorber la luz de gas.
—Un artefacto —murmuró Zev, tomándola con manos manchadas de pólvora—. Está casi seco. Apenas le queda un rastro de energía. Basura bendecida. Guárdala antes de que el metal viejo te corrompa la carne. Es la forma más rápida de volverse un Resonante, muchacho. Los artefactos son fragmentos de algo más viejo y hambriento que nosotros. Algunos te dan poder, otros solo te pudren.
Zev le devolvió la daga con un gesto de desdén y se sentó en el borde de la mesa, cruzando los brazos sobre su delantal de cuero.
—Lo que sí puedo decirte es que allá afuera no hay cuentos de hadas ni dragones. Hay atrocidades. —Zev bajó la voz, mirando fijamente la pared—. Humanos deformes, masas de carne que solo quieren consumir energía... e incluso cambia-formas. Cosas que usan la cara de tu madre o de tu hermano para acercarse y arrancarte la garganta.
Zev miró de reojo a Kael. El silencio que siguió fue asfixiante. Drack notó cómo los nudillos del subcapitán se ponían blancos.
—Así es —respondió Kael, cuya voz había perdido de golpe toda su burla habitual—. Son los mismos que se llevaron a mi familia.
—Bien, Drack —interrumpió Kael rápidamente, rompiendo el trance—. Deja de mirar esa daga. Ese metal solo sabe cortar. Ponte el uniforme. Te veré en tu habitación en cinco minutos.
Zev le lanzó un objeto metálico a Drack antes de irse: un purificador de aire que parecía un pomo de bronce.
—Pégalo a la pared, Joven Vans. Dos palmadas para activarlo. O tus pulmones se volverán ceniza.
Drack se encerró en su dormitorio. Se desvistió mecánicamente y se puso el uniforme gris acero. Era áspero y olía a almacén. Pegó el purificador a la pared y dio las dos palmadas.
Clap. Clap.
El aparato vibró y empezó a "respirar", soltando un aire fino y gélido. Drack se sentó en el borde de la cama. El colchón era duro, relleno de crin barata. Miró sus propias manos, que aún temblaban ligeramente por la tensión del juramento, y luego la daga del Dios Ciego sobre la mesa de noche. El silencio del Piso -2 no era un silencio real; era el zumbido constante de la presión bajo tierra, roto únicamente por el siseo rítmico del purificador. Respiró hondo. El aire filtrado sabía a metal frío en la lengua. Habían pasado apenas unas horas desde que salió de su casa, pero la sensación de encierro ya le oprimía el pecho. Cerró los ojos, intentando que el cansancio le ganara a la ansiedad, perdiendo la noción del tiempo en su nueva soledad, hasta que un estruendo en la madera lo hizo saltar.
TOCK. TOCK. TOCK.
—¡Drack! ¿Qué demonios haces ahí dentro? —la voz de Kael goteaba impaciencia—. ¿Estás llorando por tu papi en la oscuridad o es que tus hormonas de adolescente te han ganado y te estás haciendo una paja? ¡Sal de una vez!
Drack abrió la puerta de golpe, con el rostro encendido por la rabia.
—¿Ahora qué? Creí que podría descansar un momento.
Kael lo miró de arriba abajo, ignorando su enfado.
—Ya está bien con el "Solo Drack", el uniforme te hace oficial. Pero no hay descanso. Es hora de comer.
Subieron al Piso 1. Kael explicó que Sil y Abel no estarían; estaban fuera "purgando" amenazas. Mencionó que Sil, a pesar de su frialdad, siempre se preocupaba por dejar comida lista para "sus piezas". Al llegar a la cocina, el olor a estofado inundaba el lugar. Pero no estaban solos.
Sentado a la mesa estaba Caín Thorne, un joven tan pálido que sus venas paracían ríos de tinta, jugueteando con un frasco de veneno. Y frente a él, Valeria Vane, soltándose su coleta lateral. Su belleza era insultante, irradiaba una juventud deslumbrante.
—¿Es él? —preguntó Valeria, evaluando a Drack con una sonrisa depredadora—. Sil dijo que traería sangre azul, pero no esperaba que fuera tan... tierno.
Caín ni siquiera levantó la vista. Valeria intentó tocarlo, molestándolo con caricias fingidas, pero el chico se apartaba con una indiferencia mecánica.
—Deja de acosar al personal, Valeria —dijo Kael, interponiéndose—. El chico todavía huele a inocencia, no lo traumes el primer día porque te dio hambre de carne joven.
—Oye! —rio Valeria, sin inmutarse—. Sabes que luzco mejor que cualquier debutante. Además, la carne joven es la que mejor cicatriza.
Kael soltó una carcajada áspera mientras se sentaba.
—Claro que luces bien. Ese maquillaje alquímico te quita unos quince años de encima. Pero tú y yo sabemos que estás a un par de inviernos de cumplir los cuarenta. Afloja un poco o le vas a dar un infarto al novato.
Valeria le lanzó una mirada que prometía venganza, pero antes de que pudiera responder, Elena, la secretaria, entró con un sobre en la mano.
—¿Interrumpo? —preguntó con una cortesía que sonaba a orden. Entregó la nota a Kael—. Sil dice que es urgente.
Kael leyó la nota y su rostro cambió. Miró a Drack.
—Tu entrenamiento se adelanta. Tenemos que investigar un departamento cercano. Ruidos metálicos y olor a carne podrida. Puede ser un demente o un Corrompido.
Kael fue hacia Elena y le pidió dos receptores alquímicos: pequeñas cajas de bronce y madera que vibraban a distancia. Kael le lanzó una a Zev, que acababa de entrar al comedor.
—Tómalo, viejo —dijo Kael—. Estaremos en el Distrito de los Curtidores. Si vibra, ven con el equipo pesado.
Drack se disponía a comer su postre, un pequeño flan dorado. Justo cuando iba a tomar la cuchara, Valeria se estiró y le arrebató el plato con elegancia felina.
—Esto ahora no te hace falta, cariño —dijo ella, dándole un bocado al flan y lanzándole un beso de despedida—. Necesitas el estómago ligero para lo que vas a ver. Cuida mucho ese rostro tan bonito, Drack... sería una lástima que algo le pasara en su primera salida.
Kael le dio una palmada en el hombro a Drack y lo arrastró hacia la salida.
—Vamos, novato. El olor a muerto no espera y ya te quedaste sin flan.
Drack se puso en pie, sintiendo el peso del uniforme gris y el frío de la daga en su cintura. La inercia de su vida pasada se había roto para siempre."""
    
    # Mix text: 1 repetition of the custom story + Quijote text
    mixed_train_text = custom_story_text + "\n" + quijote_text[:split_idx]
    
    # 2. Train Tokenizer on Spanish text
    print("\nTraining BPE Tokenizer (8K vocab) on Mixed Spanish Text (Custom Story + Don Quijote)...")
    tokenizer = Tokenizer()
    tokenizer.train(mixed_train_text, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_mixed_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"BPE Tokenizer trained. Vocab size: {vocab_size}")
    
    # 3. Save splits and tokenizer
    os.makedirs("data/splits_mixed", exist_ok=True)
    os.makedirs("data/splits_mixed_val", exist_ok=True)
    with open("data/splits_mixed/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(mixed_train_text)
    # Validation split is just Quijote validation text
    with open("data/splits_mixed_val/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    tokenizer.save("data/splits_mixed/tokenizer.json")
    tokenizer.save("data/splits_mixed_val/tokenizer.json")
    
    # 4. Config & Initialize Model
    config = M01Config(
        vocab_size=vocab_size,
        context_length=256,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        num_experts=4,
        num_shared_experts=2,
        moe_top_k=2,
        use_hybrid_attention=True,
        local_window_size=16
    )
    model = TransformerLM(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
    
    # 5. Datasets
    print("Loading datasets...")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_mixed")
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_mixed_val")
    val_dataset = TinyShakespeareDataset(val_config)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training (5000 steps)
    model.train()
    steps = 5000
    step = 0
    start_time = time.time()
    
    print(f"\nStarting GPU training for {steps} steps on Mixed Spanish Text...")
    
    done = False
    while not done:
        for x, y in train_loader:
            if step >= steps:
                done = True
                break
                
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 500 == 0:
                val_loss = evaluate_val_loss(model, val_loader, device, criterion)
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Speed: {steps_per_sec:.1f} st/s | Time: {elapsed:.1f}s")
                
            step += 1
            
    print(f"\nGPU Training completed in {time.time() - start_time:.2f} seconds!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/mixed_8k.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "n_layers": config.n_layers,
            "num_experts": config.num_experts,
            "num_shared_experts": config.num_shared_experts,
            "moe_top_k": config.moe_top_k,
            "use_hybrid_attention": config.use_hybrid_attention,
            "local_window_size": config.local_window_size
        }
    }, checkpoint_path)
    print(f"Mixed Spanish checkpoint saved to {checkpoint_path}\n")
    
    # 7. Generation Queries (Verification)
    model.eval()
    
    print("-" * 50)
    prompt1 = "¿Dónde está el Quijote?"
    print(f"Generating for prompt: '{prompt1}'")
    ans1 = generate(model, tokenizer, prompt1, max_gen_len=40, temperature=0.6, device=device)
    print(ans1)
    print("-" * 50)
    
    prompt2 = "¿Dónde está Drack?"
    print(f"Generating for prompt: '{prompt2}'")
    ans2 = generate(model, tokenizer, prompt2, max_gen_len=40, temperature=0.6, device=device)
    print(ans2)
    print("-" * 50)

if __name__ == "__main__":
    main()
