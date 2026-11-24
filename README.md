# TFI — Probabilidad y Estadística con datos reales

Tecnicatura Universitaria en Programación · UTN F.R. Resistencia
Integrantes: Gomez Lara, Ferrara Sebastian, Araujo Martin, D'Aveta Tatiana Profesor: Lic. Walter N. Ramírez

## Ficha de la base

|Campo|Dato|
|-|-|
|Integrantes|*Gomez Lara, Araujo Martin, Ferrara Sebastian, D'Aveta Tatiana*|
|Fuente (enlace)|*https://www.kaggle.com/datasets/thebumpkin/800-classic-alt-rock-tracks-with-spotify-data*|
|Licencia|*MIT*|
|Fecha de descarga|*1 septiembre 2026*|
|Unidad de análisis|Una canción (track) de un artista|
|Filas × columnas|99 × 18 (original) · N final de `df\_pob`: *99*|

### Roles de las variables

|Rol|Tipo|Variable|Justificación|
|-|-|-|-|
|X|Cuantitativa continua|`Duracion\_min`|`Duration` (ms) / 60000|
|W|Cuantitativa discreta|`n\_palabras`|palabras del título (sin sufijos de versión)|
|G|Cualitativa nominal|`Modo`|`Mode`: 0 = Menor, 1 = Mayor|
|O|Cualitativa ordinal|`nivel\_energia` (y `o\_3`)|cuartiles de `Energy`|
|B|Evento binario|`muy\_popular`|`Popularity > 70` (proporción ≈ 15,2%)|
|U|Unidad de agrupamiento|`Artist`|13 artistas|

**Pregunta de interés sobre B:** ¿El modo de una canción (mayor o menor) cambia

la probabilidad de que sea muy popular (Popularity > 70)? Respuesta: Para determinar si el modo de una canción influye en la probabilidad de que sea muy popular, se compararon las probabilidades condicionales P(B|G) para los modos MAYOR y MENOR, utilizando la tabla de contingencia G × B de la sección 3.2.



La probabilidad de que una canción sea muy popular en modo MENOR es del \*\*17,39%\*\*, mientras que en modo MAYOR es del \*\*14,47%\*\*.



Por lo tanto, se observa una pequeña diferencia entre ambos modos musicales. En la población analizada, las canciones en modo MENOR presentan una probabilidad ligeramente mayor de ser muy populares. Esto sugiere que existe una asociación entre el modo musical y la popularidad.

## Semilla

`SEMILLA = 31276` rng= np.random.default\_rng(SEMILLA)`.

## Estructura

```
datos/        base\_original.xlsx · df\_pob.csv · parametros.json
notebooks/    tfi\_grupogafd.ipynb
figuras/      gráficos exportados a 150 dpi
funciones.py  preparar, media, varianza\_pob, mediana, sturges, bayes, dibujar\_arbol, pmf\_\*
requirements.txt
README.md
```

